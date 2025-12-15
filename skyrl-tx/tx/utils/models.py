from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from cloudpathlib import CloudPath
from flax import nnx
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
import numpy as np
import optax
import safetensors
import safetensors.numpy
from transformers import PretrainedConfig
from tqdm import tqdm
import peft

from tx.utils.log import logger
from tx.utils.storage import download_and_unpack, pack_and_upload
from tx.tinker.types import LoraConfig

if TYPE_CHECKING:
    import torch


def resolve_model_path(model_name_or_path: str) -> str:
    """Resolve a model name or path to a local directory path.

    If the model_name_or_path points to an existing local directory, it will be
    used directly. Otherwise, the model will be downloaded from HuggingFace Hub.

    Args:
        model_name_or_path: Either a local path to a model directory or a
            HuggingFace model identifier (e.g., "Qwen/Qwen3-0.6B").

    Returns:
        Path to the local directory containing model config and weights.
    """
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        logger.info(f"Using local model at {local_path}")
        return str(local_path)
    return snapshot_download(model_name_or_path, allow_patterns=["*.safetensors", "*.json"])


def get_dtype(dtype: str | torch.dtype) -> jnp.dtype:
    "Convert torch dtype to jax dtype."

    match str(dtype):
        case "torch.float32" | "float32":
            return jnp.float32
        case "torch.bfloat16" | "bfloat16":
            return jnp.bfloat16
        case "torch.float16" | "float16":
            return jnp.float16
        case _:
            raise ValueError(f"Unsupported torch dtype: {dtype}")


def get_model_class(config: PretrainedConfig) -> Callable[..., nnx.Module]:
    "Get the correct model class based on the config."
    import tx.models.qwen3

    for architecture in config.architectures or []:
        if hasattr(tx.models.qwen3, architecture):
            return getattr(tx.models.qwen3, architecture)

    raise ValueError(f"None of the architectures {config.architectures} is currently supported.")


def get_param_key(path: tuple, prefix: str = "") -> str:
    "Get the safetensors key for a given model path."
    if path[-1] in {"embedding", "kernel"}:
        path = (*path[:-1], "weight")
    elif path[-1] in {"lora_A", "lora_B"}:
        path = (*path, "weight")
    return prefix + ".".join(map(str, path))


def is_scanned_layer_param(path: tuple) -> bool:
    """Check if a param path is from a scanned layers module (no layer index)."""
    # With scan_layers=True, paths look like ('model', 'layers', 'self_attn', ...)
    # Without scan, paths have layer index: ('model', 'layers', '0', 'self_attn', ...)
    if "layers" not in path:
        return False
    layers_idx = path.index("layers")
    # If next element after 'layers' is not a digit, it's scanned
    if layers_idx + 1 < len(path):
        next_elem = path[layers_idx + 1]
        return not (isinstance(next_elem, int) or (isinstance(next_elem, str) and next_elem.isdigit()))
    return False


def get_layer_param_key(path: tuple, layer_idx: int, prefix: str = "") -> str:
    """Get the safetensors key for a scanned layer param at a specific layer index."""
    # Insert layer index after 'layers' in path
    if path[-1] in {"embedding", "kernel"}:
        path = (*path[:-1], "weight")
    elif path[-1] in {"lora_A", "lora_B"}:
        path = (*path, "weight")

    layers_idx = path.index("layers")
    new_path = path[:layers_idx + 1] + (str(layer_idx),) + path[layers_idx + 1:]
    return prefix + ".".join(map(str, new_path))


def get_expert_key(path: tuple, expert_idx: int, layer_idx: int | None = None) -> str:
    """Get the safetensors key for an expert weight model path.

    If layer_idx is provided, inserts layer index after 'layers' (for scanned models).
    """
    if path[-1] in {"embedding", "kernel"}:
        path = (*path[:-1], "weight")
    else:
        # For nnx.Param directly on module, append weight
        path = (*path, "weight")
    path = tuple(s if s != "experts" else f"experts.{expert_idx}" for s in path)

    if layer_idx is not None and "layers" in path:
        layers_idx = path.index("layers")
        path = path[:layers_idx + 1] + (str(layer_idx),) + path[layers_idx + 1:]

    return ".".join(map(str, path))


def load_safetensors(
    checkpoint_dir: str | os.PathLike,
    config: PretrainedConfig,
    model: nnx.Module,
    skip_lora: bool = True,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
    reshape_for_scan: bool = False,
) -> None:
    """Load safetensors with streaming to reduce peak memory usage.

    Uses safetensors.safe_open for lazy loading - tensors are only loaded into
    memory when accessed. Updates model parameters incrementally to minimize
    device memory usage (only 1 extra tensor on device at a time).

    Args:
        reshape_for_scan: If True and segment_length is set, reshape stacked layer
            weights from [num_layers, ...] to [num_segments, segment_length, ...].
            Set to False if nnx.scan handles the reshaping internally.
    """
    # Build index: map each key to its file handle (lazy loading)
    file_handles = []
    key_to_handle = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        handle = safetensors.safe_open(file, framework="numpy")
        file_handles.append(handle)
        for key in handle.keys():
            clean_key = key.removeprefix(prefix)
            key_to_handle[clean_key] = (handle, key)

    def get_tensor(key: str) -> np.ndarray:
        """Load a single tensor on demand."""
        if key not in key_to_handle:
            raise KeyError(f"Key {key} not found in checkpoint")
        handle, orig_key = key_to_handle[key]
        return handle.get_tensor(orig_key)

    # Check if model uses scan_layers and segment_length
    scan_layers = getattr(config, "scan_layers", False)
    num_layers = getattr(config, "num_hidden_layers", None)
    segment_length = getattr(config, "segment_length", None)
    use_fused_moe = getattr(config, "use_fused_moe", False)

    def maybe_reshape_to_segments(tensor: np.ndarray) -> np.ndarray:
        """Reshape tensor from [num_layers, ...] to [num_segments, segment_length, ...] if segment_length is set."""
        if reshape_for_scan and segment_length is not None and tensor.shape[0] == num_layers:
            num_segments = num_layers // segment_length
            new_shape = (num_segments, segment_length) + tensor.shape[1:]
            return tensor.reshape(new_shape)
        return tensor

    model_params = nnx.to_flat_state(nnx.state(model))
    for path, param in tqdm(model_params, desc="Loading params"):
        if filter_fn is not None and not filter_fn(path):
            continue
        # Skip LoRA parameters if requested
        if skip_lora and ("lora_a" in path or "lora_b" in path or "lora_A" in path or "lora_B" in path or "lora_scaling" in path or "lora_ranks" in path or "lora" in path):
            continue

        # Load tensor on demand
        if "experts" in path:
            # Check if this is fused_moe format (w1/w2) or standard format (gate_proj/up_proj/down_proj)
            is_fused_w1 = use_fused_moe and "w1" in path
            is_fused_w2 = use_fused_moe and "w2" in path

            if is_fused_w1:
                # w1 combines gate_proj and up_proj: (E, I*2, H)
                # Checkpoint has gate_proj: (I, H) and up_proj: (I, H) per expert
                # Need to load both, transpose to (H, I), then transpose to (I, H) for fused format
                def load_w1_for_layer(layer_idx=None):
                    gate_path = tuple(s if s != "w1" else "gate_proj" for s in path)
                    up_path = tuple(s if s != "w1" else "up_proj" for s in path)
                    gate_tensors = []
                    up_tensors = []
                    for i in range(config.num_experts):
                        gate_key = get_expert_key(gate_path, i, layer_idx)
                        up_key = get_expert_key(up_path, i, layer_idx)
                        # Checkpoint: (I, H), transpose to (H, I), then transpose to (I, H) for fused
                        gate_tensors.append(get_tensor(gate_key))  # (I, H)
                        up_tensors.append(get_tensor(up_key))  # (I, H)
                    # Stack: (E, I, H), concatenate gate+up on axis 1: (E, I*2, H)
                    gate_stacked = np.stack(gate_tensors, axis=0)  # (E, I, H)
                    up_stacked = np.stack(up_tensors, axis=0)  # (E, I, H)
                    return np.concatenate([gate_stacked, up_stacked], axis=1)  # (E, I*2, H)

                if scan_layers and is_scanned_layer_param(path):
                    layer_tensors = [load_w1_for_layer(layer_idx) for layer_idx in range(num_layers)]
                    tensor = np.stack(layer_tensors, axis=0)
                    tensor = maybe_reshape_to_segments(tensor)
                    del layer_tensors
                else:
                    tensor = load_w1_for_layer()

            elif is_fused_w2:
                # w2 is transposed down_proj: (E, H, I)
                # Checkpoint has down_proj: (H, I) per expert
                def load_w2_for_layer(layer_idx=None):
                    down_path = tuple(s if s != "w2" else "down_proj" for s in path)
                    down_tensors = []
                    for i in range(config.num_experts):
                        down_key = get_expert_key(down_path, i, layer_idx)
                        # Checkpoint: (H, I), we need (E, H, I)
                        down_tensors.append(get_tensor(down_key))  # (H, I)
                    return np.stack(down_tensors, axis=0)  # (E, H, I)

                if scan_layers and is_scanned_layer_param(path):
                    layer_tensors = [load_w2_for_layer(layer_idx) for layer_idx in range(num_layers)]
                    tensor = np.stack(layer_tensors, axis=0)
                    tensor = maybe_reshape_to_segments(tensor)
                    del layer_tensors
                else:
                    tensor = load_w2_for_layer()

            elif scan_layers and is_scanned_layer_param(path):
                # For scanned layers with experts: load per-layer, per-expert tensors
                # Shape will be [num_layers, num_experts, ...]
                layer_tensors = []
                for layer_idx in range(num_layers):
                    expert_tensors = [get_tensor(get_expert_key(path, i, layer_idx)).T for i in range(config.num_experts)]
                    layer_tensors.append(np.stack(expert_tensors, axis=0))
                    del expert_tensors
                tensor = np.stack(layer_tensors, axis=0)
                tensor = maybe_reshape_to_segments(tensor)
                del layer_tensors
            else:
                expert_tensors = [get_tensor(get_expert_key(path, i)).T for i in range(config.num_experts)]
                tensor = np.stack(expert_tensors, axis=0)
                # No segment reshape for non-scanned experts
                del expert_tensors
        elif scan_layers and is_scanned_layer_param(path):
            # For scanned layers: nnx.vmap stacks weights during init, but checkpoint
            # has individual layer weights. Load per-layer tensors and stack them.
            layer_tensors = []
            for layer_idx in range(num_layers):
                layer_key = get_layer_param_key(path, layer_idx, prefix)
                layer_tensor = get_tensor(layer_key)
                layer_tensor = layer_tensor if "embed_tokens" in path else layer_tensor.T
                if path[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                    # Reshape attention projections per-layer before stacking
                    # If reshape_for_scan is enabled, param.shape is [num_segments, segment_length, ...], skip 2 dims
                    # Otherwise param.shape is [num_layers, ...], skip 1 dim
                    skip_dims = 2 if reshape_for_scan and segment_length is not None else 1
                    target_shape = param.shape[skip_dims:]
                    layer_tensor = layer_tensor.reshape(target_shape)
                layer_tensors.append(layer_tensor)
            tensor = np.stack(layer_tensors, axis=0)
            tensor = maybe_reshape_to_segments(tensor)
            del layer_tensors
        else:
            key = get_param_key(path)
            tensor = get_tensor(key)
            tensor = tensor if "embed_tokens" in path else tensor.T
            if path[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                tensor = tensor.reshape(param.shape)

        assert param.shape == tensor.shape, f"shape mismatch for {path}: expected {param.shape}, got {tensor.shape}"
        sharded_tensor = jax.device_put(tensor.astype(param.dtype), param.sharding)

        # Update model incrementally - only 1 extra tensor on device at a time
        nnx.update(model, nnx.from_flat_state([(path, sharded_tensor)]))

        # Explicit cleanup to free host memory immediately
        del tensor
        del sharded_tensor

    # Close file handles
    file_handles.clear()


def save_safetensors(
    config: PretrainedConfig,
    model: nnx.Module,
    filename: Path,
    prefix: str = "",
    filter_fn: Callable[[tuple], bool] | None = None,
) -> None:
    model_params = nnx.to_flat_state(nnx.state(model))
    tensors = {}
    for path, param in model_params:
        if "rngs" in path:
            continue
        if filter_fn is not None and not filter_fn(path):
            continue
        key = get_param_key(path, prefix=prefix)
        if "experts" in path:
            for i in range(config.num_experts):
                tensors[get_expert_key(path, i)] = param[i, :, :].T
            continue
        if "q_proj" in path or "k_proj" in path or "v_proj" in path:
            param = param.reshape(param.shape[0], -1)
        elif "o_proj" in path:
            param = param.reshape(-1, param.shape[-1])
        tensors[key] = param if "embed_tokens" in path else param.T
    safetensors.numpy.save_file(tensors, filename)


def filter_lora(adapter_config: LoraConfig, path: tuple[str, ...]) -> bool:
    if not adapter_config.train_attn and "self_attn" in path:
        return False
    if not adapter_config.train_mlp and ("mlp" in path or "experts" in path):
        return False
    if not adapter_config.train_unembed and ("embed_tokens" in path or "lm_head" in path):
        return False
    return True


def load_lora_checkpoint(
    model: nnx.Module, adapter_config: LoraConfig, adapter_index: int, checkpoint_path: Path | CloudPath
) -> None:
    """Load LoRA adapter weights from a sampling checkpoint into the model.

    Args:
        model: The Qwen3ForCausalLM model to load the adapter into
        adapter_config: LoRA adapter configuration
        adapter_index: Index of the adapter to load into
        checkpoint_path: Path to the checkpoint tar.gz file
    """
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    adapter_lora_params = extract_adapter_state(adapter_index, lora_params, adapter_config.rank)

    with download_and_unpack(checkpoint_path) as temp_dir:
        load_safetensors(
            temp_dir,
            model.config,
            adapter_lora_params,
            skip_lora=False,
            prefix="base_model.model.",
            filter_fn=lambda path: filter_lora(adapter_config, path),
        )
    insert_adapter_state(adapter_index, lora_params, adapter_lora_params, adapter_config.rank)


def save_lora_checkpoint(
    model: nnx.Module,
    base_model_name: str,
    adapter_config: LoraConfig,
    adapter_index: int,
    output_path: Path | CloudPath,
):
    """Save a LoRA checkpoint as a tar.gz archive.

    Args:
        model: The Qwen3ForCausalLM model to extract LoRA parameters from
        adapter_config: LoRA adapter configuration
        adapter_index: Index of the adapter to save
        output_path: Path to save the checkpoint tar.gz file
    """
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    adapter_lora_params = extract_adapter_state(adapter_index, lora_params, adapter_config.rank)

    peft_config = peft.LoraConfig(
        base_model_name_or_path=base_model_name, r=adapter_config.rank, lora_alpha=adapter_config.alpha
    )

    with pack_and_upload(output_path) as temp_dir:
        save_safetensors(
            model.config,
            adapter_lora_params,
            temp_dir / "adapter_model.safetensors",
            prefix="base_model.model.",
            filter_fn=lambda path: filter_lora(adapter_config, path),
        )
        peft_config.save_pretrained(temp_dir)


class OptimizerName(str, Enum):
    adamw = "adamw"


def get_optimizer(optimizer_name: OptimizerName, optimizer_args: dict) -> optax.GradientTransformation:
    match (optimizer_name, optimizer_args):
        case (OptimizerName.adamw, {"learning_rate": lr, **kwargs}):
            return optax.adamw(lr, **kwargs)
        case (_, {"learning_rate": _}):
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        case _:
            raise ValueError("The 'learning_rate' key must be provided in optimizer_args.")


@nnx.jit(static_argnames=("adapter_index", "rank"))
def extract_adapter_state(adapter_index: int, lora_params: nnx.GraphState, rank: int) -> nnx.GraphState:
    "Helper function to extract the adapter parameters for a specific adapter index."

    def extract_state(path: tuple, p: jnp.ndarray):
        if path[-2].key not in {"lora_A", "lora_B"}:
            return p
        assert p.ndim in {3, 4}, f"LoRA parameters must have 3 or 4 dimensions, got shape {p.shape}"
        if path[-2].key == "lora_A":
            return p[adapter_index, ..., :, :rank]
        if path[-2].key == "lora_B":
            return p[adapter_index, ..., :rank, :]

    return jax.tree.map_with_path(extract_state, lora_params)


# We need to use nnx.jit here instead of jax.jit so the nnx.update will be handled correctly
@nnx.jit(static_argnames=("adapter_index", "rank"))
def insert_adapter_state(
    adapter_index: int, lora_params: nnx.GraphState, new_params: nnx.GraphState, rank: int
) -> None:
    "Helper function to insert the adapter parameters for a specific adapter index (inverse of extract_adapter_state)."

    def insert_state(path: tuple, p: jax.Array, new: jax.Array):
        if path[-2].key not in {"lora_A", "lora_B"}:
            return new
        assert p.ndim in {3, 4}, f"LoRA parameters must have 3 or 4 dimensions, got shape {p.shape}"
        if path[-2].key == "lora_A":
            return p.at[adapter_index, ..., :, :rank].set(new)
        elif path[-2].key == "lora_B":
            return p.at[adapter_index, ..., :rank, :].set(new)

    updated = jax.tree.map_with_path(insert_state, lora_params, new_params)
    nnx.update(lora_params, updated)


def convert_maxtext_lora_to_hf(
    lora_state: nnx.State,
    output_path: Path,
    base_model_name: str = "",
    lora_rank: int = 8,
    lora_alpha: int = 32,
) -> None:
    """Convert MaxText LoRA tensors to HuggingFace PEFT format.

    MaxText LoRA shapes (layer axis in middle, heads sometimes factored):
        - query/key/value lora_a: (hidden_size, num_layers, rank)
        - query lora_b: (rank, num_layers, num_heads, head_dim)
        - key/value lora_b: (rank, num_layers, num_kv_heads, head_dim)
        - out lora_a: (num_heads, num_layers, head_dim, rank)
        - out lora_b: (rank, num_layers, hidden_size)

    HuggingFace PEFT format (per layer):
        - base_model.model.model.layers.{i}.self_attn.{proj}.lora_A.weight: (rank, in_features)
        - base_model.model.model.layers.{i}.self_attn.{proj}.lora_B.weight: (out_features, rank)

    Args:
        lora_state: NNX state containing MaxText LoRA parameters
        output_path: Path to save the PEFT checkpoint (directory)
        base_model_name: Name of the base model for PEFT config
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
    """
    # Map MaxText projection names to HuggingFace names
    proj_name_map = {
        "query": "q_proj",
        "key": "k_proj",
        "value": "v_proj",
        "out": "o_proj",
    }

    # Collect all LoRA tensors by path
    lora_tensors = {}
    for path, val in jax.tree_util.tree_leaves_with_path(lora_state):
        path_str = "/".join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
        lora_tensors[path_str] = np.asarray(val)

    # Determine num_layers from any tensor (layer axis is at position 1 for most)
    sample_tensor = next(iter(lora_tensors.values()))
    # Find layer dimension - it's 48 in the examples
    num_layers = None
    for tensor in lora_tensors.values():
        # Layer axis is typically at position 1 for lora_a, position 1 for lora_b
        if tensor.ndim >= 2 and tensor.shape[1] == 48:
            num_layers = 48
            break
        if tensor.ndim >= 2 and tensor.shape[0] == 48:
            num_layers = 48
            break

    if num_layers is None:
        # Try to infer from shapes
        for tensor in lora_tensors.values():
            for dim in tensor.shape:
                if dim in [36, 48, 64, 80, 96]:  # Common layer counts
                    num_layers = dim
                    break
            if num_layers:
                break

    if num_layers is None:
        raise ValueError("Could not determine num_layers from tensor shapes")

    logger.info(f"Converting MaxText LoRA to HuggingFace format: {num_layers} layers, rank={lora_rank}")

    # Output tensors in HuggingFace format
    hf_tensors = {}

    for path_str, tensor in lora_tensors.items():
        # Parse the path to identify projection and lora_a/lora_b
        # Example: base/decoder/layers/self_attention/query/lora_a/.value
        parts = path_str.split("/")

        # Find projection name and lora type
        proj_name = None
        lora_type = None
        for i, part in enumerate(parts):
            if part in proj_name_map:
                proj_name = proj_name_map[part]
            if part in ("lora_a", "lora_b"):
                lora_type = "lora_A" if part == "lora_a" else "lora_B"

        if proj_name is None or lora_type is None:
            logger.warning(f"Skipping unrecognized path: {path_str}")
            continue

        logger.info(f"Converting {path_str}: shape {tensor.shape} -> {proj_name}/{lora_type}")

        # Convert based on projection and lora type
        # MaxText shapes vary, need to handle each case
        for layer_idx in range(num_layers):
            if lora_type == "lora_A":
                if proj_name == "o_proj":
                    # out lora_a: (num_heads, num_layers, head_dim, rank) -> (rank, num_heads * head_dim)
                    # Layer axis at position 1
                    layer_tensor = tensor[:, layer_idx, :, :]  # (num_heads, head_dim, rank)
                    # Flatten heads: (num_heads * head_dim, rank) then transpose to (rank, in_features)
                    layer_tensor = layer_tensor.reshape(-1, layer_tensor.shape[-1])  # (in_features, rank)
                    layer_tensor = layer_tensor.T  # (rank, in_features)
                else:
                    # query/key/value lora_a: (hidden_size, num_layers, rank) -> (rank, hidden_size)
                    # Layer axis at position 1
                    layer_tensor = tensor[:, layer_idx, :]  # (hidden_size, rank)
                    layer_tensor = layer_tensor.T  # (rank, hidden_size)
            else:  # lora_B
                if proj_name == "o_proj":
                    # out lora_b: (rank, num_layers, hidden_size) -> (hidden_size, rank)
                    # Layer axis at position 1
                    layer_tensor = tensor[:, layer_idx, :]  # (rank, hidden_size)
                    layer_tensor = layer_tensor.T  # (hidden_size, rank)
                else:
                    # query lora_b: (rank, num_layers, num_heads, head_dim) -> (num_heads * head_dim, rank)
                    # key/value lora_b: (rank, num_layers, num_kv_heads, head_dim) -> (num_kv_heads * head_dim, rank)
                    # Layer axis at position 1
                    layer_tensor = tensor[:, layer_idx, ...]  # (rank, num_heads, head_dim) or (rank, num_kv_heads, head_dim)
                    # Flatten heads and transpose: (rank, out_features) -> (out_features, rank)
                    layer_tensor = layer_tensor.reshape(layer_tensor.shape[0], -1)  # (rank, out_features)
                    layer_tensor = layer_tensor.T  # (out_features, rank)

            # HuggingFace PEFT key format
            hf_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj_name}.{lora_type}.weight"
            hf_tensors[hf_key] = layer_tensor.astype(np.float32)

    # Save as safetensors
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    safetensors.numpy.save_file(hf_tensors, output_path / "adapter_model.safetensors")
    logger.info(f"Saved {len(hf_tensors)} tensors to {output_path / 'adapter_model.safetensors'}")

    # Save PEFT config
    peft_config = peft.LoraConfig(
        base_model_name_or_path=base_model_name,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    peft_config.save_pretrained(output_path)
    logger.info(f"Saved PEFT config to {output_path}")


def round_up_seq_len(seq_len: int, min_seq_len: int = 32) -> int:
    """
    Rounds a sequence length up to roughly two significant binary digits.
    We do this to pad sequences, so the Jax JIT compiler needs to
    compile fewer different shapes.
    """
    if seq_len <= min_seq_len:
        return min_seq_len

    # Find the position of the most significant bit.
    msb_pos = seq_len.bit_length() - 1
    # Create a mask for the two most significant bits.
    mask = (1 << msb_pos) | (1 << (msb_pos - 1))
    # Round down to the nearest value with at most two significant bits.
    result = seq_len & mask

    # If we rounded down, round up to the next bucket boundary.
    if result < seq_len:
        result += 1 << (msb_pos - 1)

    return result

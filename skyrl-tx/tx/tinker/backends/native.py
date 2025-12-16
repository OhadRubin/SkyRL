"""Native LoRA backend for TinkerEngine (Qwen3 + LoRA)."""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.training import checkpoints
from transformers import PretrainedConfig

from tx.models.configs import Qwen3Config
from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.backends.backend import AbstractBackend
from tx.tinker.loss_fns import LOSS_TYPES, LOSS_FUNCTIONS
from tx.utils.models import (
    get_dtype,
    get_model_class,
    load_safetensors,
    load_lora_checkpoint,
    save_lora_checkpoint,
    extract_adapter_state,
    insert_adapter_state,
    round_up_seq_len,
    resolve_model_path,
)
from tx.utils.storage import pack_and_upload
from tx.utils.log import logger


def pad(xs, pad_to: int, *, fill):
    """Pad a list to a specified length with a fill value."""
    return xs + ([fill] * (pad_to - len(xs)))


def pad_batch(sequences: list[list], max_length: int, dtype, left: bool = False) -> jax.Array:
    """Pad a batch of sequences to max_length."""
    batch_size = len(sequences)
    padded = np.zeros((batch_size, max_length), dtype=dtype)
    for i, seq in enumerate(sequences):
        assert len(seq) <= max_length, f"Sequence length {len(seq)} exceeds max_length {max_length}"
        if left:
            padded[i, max_length - len(seq):] = seq
        else:
            padded[i, :len(seq)] = seq
    return jnp.asarray(padded)


def _count_params(pytree) -> int:
    """Count total number of parameters in a pytree."""
    def get_numel(x):
        if hasattr(x, 'shape'):
            return int(np.prod(x.shape))
        return 0
    counts = jax.tree.leaves(jax.tree.map(get_numel, pytree))
    return sum(counts)


@jax.tree_util.register_dataclass
@dataclass
class AccumulatedGradients:
    """Stores accumulated gradients."""

    grad_sum: nnx.State
    count: jax.Array

    @classmethod
    def create(cls, lora_params: nnx.State) -> "AccumulatedGradients":
        """Initialize with zeros."""
        return cls(
            grad_sum=jax.tree.map(jnp.zeros_like, lora_params),
            count=jnp.zeros((1,), dtype=jnp.int32),
        )

    def add(self, lora_grads: nnx.State, batch_size: int) -> "AccumulatedGradients":
        """Accumulate gradients and increment count."""
        return AccumulatedGradients(
            grad_sum=jax.tree.map(lambda a, b: a + b, self.grad_sum, lora_grads),
            count=self.count + batch_size,
        )

    def get_mean(self) -> nnx.State:
        """Compute mean gradients."""
        return jax.tree.map(
            lambda g: g / self.count.astype(g.dtype),
            self.grad_sum,
        )

    def reset(self) -> "AccumulatedGradients":
        """Reset gradients and count."""
        return AccumulatedGradients(
            grad_sum=jax.tree.map(jnp.zeros_like, self.grad_sum),
            count=jnp.zeros((1,), dtype=jnp.int32),
        )


class NativeBackend(AbstractBackend):
    """Backend for Qwen3 models with LoRA adapters."""

    def __init__(self, config: EngineConfig):
        """Initialize Native LoRA backend."""
        self.config = config
        self.metrics = types.EngineMetrics()

        # Initialize the shared base model with LoRA config
        checkpoint_path = resolve_model_path(config.base_model)
        base_config = PretrainedConfig.from_pretrained(checkpoint_path)
        self.model_config = Qwen3Config(
            base_config,
            max_lora_rank=config.max_lora_rank,
            shard_attention_heads=config.shard_attention_heads,
            mlp_lora=config.mlp_lora,
            attn_lora=config.attn_lora,
            embed_lora=config.embed_lora,
            scan_layers=config.scan_layers,
            segment_length=config.segment_length,
            use_ring_attention=config.use_ring_attention,
            scan_query_chunk_size=config.scan_query_chunk_size,
            scan_key_chunk_size=config.scan_key_chunk_size,
            use_fused_moe=config.use_fused_moe,
            use_maxtext_moe=config.use_maxtext_moe,
        )

        model_class = get_model_class(self.model_config)

        # Create model and load weights
        self.mesh = jax.make_mesh((1, 1, config.tensor_parallel_size), ("layer", "dp", "tensor"))
        with jax.set_mesh(self.mesh):
            self.model = model_class(
                self.model_config,
                dtype=get_dtype(self.model_config.dtype),
                rngs=nnx.Rngs(0),
                mesh=self.mesh
            )
            if config.load_safetensors:
                load_safetensors(
                    checkpoint_path,
                    self.model_config,
                    self.model,
                    reshape_for_scan=self.model_config.reshape_for_scan
                )

            # Split model into LoRA and non-LoRA parameters
            self.graphdef, self.lora_params, self.non_lora_params = nnx.split(
                self.model, self.model.is_lora_param, ...
            )

            # Initialize accumulated gradients
            self.accumulated_grads = AccumulatedGradients.create(self.lora_params)
            self._log_accumulated_grads()

        logger.info(
            f"Initialized base model {config.base_model} with "
            f"max_lora_adapters={config.max_lora_adapters}, max_lora_rank={config.max_lora_rank}"
        )

        self._create_loss_and_grad_fn()

    def _log_accumulated_grads(self):
        """Log accumulated gradient structure."""
        accum_params = _count_params(self.accumulated_grads.grad_sum)
        logger.info(f"[LoRA] Accumulated grads total params: {accum_params / 1e6:.2f}M")
        for path, val in jax.tree_util.tree_leaves_with_path(self.accumulated_grads.grad_sum):
            path_str = "/".join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
            logger.info(f"  {path_str}: {val.shape}")

    def _micro_batch_size(self, total: int) -> int:
        """Return effective micro-batch size."""
        mb = self.config.train_micro_batch_size
        return total if mb <= 0 else max(1, min(mb, total))

    @contextmanager
    def _jit_timing_context(self, seq_len: int, mode: str):
        """Context manager to track JIT compilation times."""
        jit_times = self.metrics.train_seq_len_jit_times if mode == "train" else self.metrics.sample_seq_len_jit_times
        if not self.config.enforce_eager and seq_len not in jit_times:
            logger.info(f"JIT compiling for {mode} seq_len={seq_len} in progress...")
            start_time = time.time()
            yield
            elapsed = time.time() - start_time
            jit_times[seq_len] = elapsed
            logger.info(f"JIT compilation for {mode} seq_len={seq_len} took {elapsed:.2f}s")
        else:
            yield

    def _create_loss_and_grad_fn(self):
        """Create loss and gradient functions for Qwen3+LoRA model."""
        def loss_for_model(
            model,
            input_ids: jax.Array,
            attention_mask: jax.Array,
            target_ids: jax.Array,
            loss_mask: jax.Array,
            loss_fn_types: jax.Array,
            sampling_logprobs: jax.Array,
            advantages: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits

            logprobs = jax.nn.log_softmax(logits, axis=-1)
            target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)

            def compute_loss_per_example(loss_fn_type, target_logprobs, loss_mask, sampling_logprobs, advantages):
                return jax.lax.switch(
                    loss_fn_type,
                    LOSS_FUNCTIONS,
                    target_logprobs,
                    loss_mask,
                    sampling_logprobs,
                    advantages,
                )

            per_token_losses = jax.vmap(compute_loss_per_example)(
                loss_fn_types, target_logprobs, loss_mask, sampling_logprobs, advantages,
            )

            per_seq_loss = per_token_losses.sum(axis=-1) / loss_mask.sum(axis=-1)
            return per_seq_loss.sum(), (target_logprobs, per_token_losses)

        lora_filter = nnx.All(nnx.Param, nnx.Any(nnx.PathContains("lora_a"), nnx.PathContains("lora_b")))
        loss_and_grad_fn = nnx.value_and_grad(
            loss_for_model,
            argnums=nnx.DiffState(0, lora_filter),
            has_aux=True
        )

        def forward_backward_and_accumulate(
            accumulated_grads: AccumulatedGradients,
            model,
            input_ids: jax.Array,
            attention_mask: jax.Array,
            target_ids: jax.Array,
            loss_mask: jax.Array,
            loss_fn_types: jax.Array,
            sampling_logprobs: jax.Array,
            advantages: jax.Array,
        ) -> tuple[AccumulatedGradients, jax.Array, jax.Array, jax.Array]:
            """Fused forward-backward-accumulate operation."""
            (loss, (target_logprobs, per_token_losses)), lora_grads = loss_and_grad_fn(
                model, input_ids, attention_mask, target_ids, loss_mask,
                loss_fn_types, sampling_logprobs, advantages,
            )
            batch_size = input_ids.shape[0]
            new_accumulated_grads = accumulated_grads.add(lora_grads, batch_size)
            return new_accumulated_grads, per_token_losses, target_logprobs, loss

        if self.config.enforce_eager:
            self._forward_backward_and_accumulate = forward_backward_and_accumulate
        else:
            lora_shardings = jax.tree.map(
                lambda x: jax.NamedSharding(self.mesh, x.sharding.spec), self.lora_params
            )
            accumulated_grads_shardings = AccumulatedGradients(
                grad_sum=lora_shardings,
                count=jax.NamedSharding(self.mesh, jax.P(None)),
            )
            replicated = jax.NamedSharding(self.mesh, jax.P(None))
            scalar = jax.NamedSharding(self.mesh, jax.P())

            self._forward_backward_and_accumulate = jax.jit(
                forward_backward_and_accumulate,
                in_shardings=(accumulated_grads_shardings, None, replicated, replicated, replicated, replicated, replicated, replicated, replicated),
                out_shardings=(accumulated_grads_shardings, replicated, replicated, scalar),
                donate_argnames=("accumulated_grads",),
            )

        def compute_grads_and_update(
            accumulated_grads: AccumulatedGradients,
            lora_params: nnx.State,
            optimizer: nnx.Optimizer,
            adapter_index: jax.Array,
        ) -> AccumulatedGradients:
            """Compute full gradients, apply optimizer update, and reset."""
            optimizer.update(lora_params, accumulated_grads.get_mean())
            return accumulated_grads.reset()

        if self.config.enforce_eager:
            self._compute_grads_and_update = compute_grads_and_update
        else:
            self._compute_grads_and_update = nnx.jit(compute_grads_and_update)

    def create_optimizer(self, model_id: str) -> nnx.Optimizer:
        """Create an optimizer for a model."""
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
        return nnx.Optimizer(self.model, tx, wrt=self.model.is_lora_param)

    def precompile_kernels(self, seq_lens: list[int]) -> None:
        """Precompile JIT kernels for specified sequence lengths."""
        if not seq_lens or self.config.enforce_eager:
            return

        logger.info(f"Precompiling JIT kernels for sequence lengths: {seq_lens}")
        micro_bs = max(1, self.config.train_micro_batch_size) if self.config.train_micro_batch_size > 0 else 1

        with jax.set_mesh(self.mesh):
            for seq_len in seq_lens:
                dummy_input_ids = jnp.zeros((micro_bs, seq_len), dtype=jnp.int32)
                dummy_target_ids = jnp.zeros((micro_bs, seq_len), dtype=jnp.int32)
                dummy_loss_mask = jnp.ones((micro_bs, seq_len), dtype=jnp.float32)
                dummy_attention_mask = jnp.ones((micro_bs, seq_len), dtype=jnp.int32)
                dummy_loss_fn_types = jnp.zeros((micro_bs,), dtype=jnp.int32)
                dummy_sampling_logprobs = jnp.zeros((micro_bs, seq_len), dtype=jnp.float32)
                dummy_advantages = jnp.zeros((micro_bs, seq_len), dtype=jnp.float32)

                with self._jit_timing_context(seq_len, mode="train"):
                    self.accumulated_grads, _, _, _ = self._forward_backward_and_accumulate(
                        self.accumulated_grads,
                        self.model,
                        dummy_input_ids,
                        dummy_attention_mask,
                        dummy_target_ids,
                        dummy_loss_mask,
                        dummy_loss_fn_types,
                        dummy_sampling_logprobs,
                        dummy_advantages,
                    )

                self.accumulated_grads = AccumulatedGradients.create(self.lora_params)

        logger.info(f"Precompilation complete for {len(seq_lens)} sequence lengths")

    def process_forward_backward_batch(
        self,
        requests: dict[str, tuple[str, types.ForwardBackwardInput]],
        models: dict[str, types.ModelMetadata],
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process multiple forward_backward requests in a single batch."""
        if not requests:
            return {}

        results = {}
        all_input_ids = []
        all_targets = []
        all_token_weights = []
        all_adapter_indices = []
        request_batch_slices = []
        all_sampling_logprobs = []
        all_advantages = []
        all_loss_fn_types = []

        for request_id, (model_id, request_data) in requests.items():
            if model_id not in models:
                results[request_id] = types.ErrorResponse(error=f"Model {model_id} not loaded", status="failed")
                continue

            adapter_index = models[model_id].adapter_index
            loss_fn_type = LOSS_TYPES[request_data.loss_fn]
            request_start = len(all_input_ids)

            for item in request_data.data:
                tokens = [t for chunk in item.model_input.chunks for t in chunk.tokens]
                all_input_ids.append(tokens)
                loss_fn_inputs = item.loss_fn_inputs
                all_targets.append(loss_fn_inputs.target_tokens.data)
                all_token_weights.append(loss_fn_inputs.weights.data)
                all_sampling_logprobs.append(loss_fn_inputs.logprobs.data)
                all_advantages.append(loss_fn_inputs.advantages.data)
                all_adapter_indices.append(adapter_index)
                all_loss_fn_types.append(loss_fn_type)

            request_batch_slices.append((request_id, model_id, request_start, len(all_input_ids)))

        if not all_input_ids:
            return results

        max_len = round_up_seq_len(max(len(seq) for seq in all_input_ids), self.config.min_seq_len)

        input_ids = pad_batch(all_input_ids, max_len, np.int32)
        target_ids = pad_batch(all_targets, max_len, np.int32)
        loss_fn_types = jnp.array(all_loss_fn_types, dtype=jnp.int32)
        attention_mask = pad_batch([[1] * len(seq) for seq in all_input_ids], max_len, np.int32)
        loss_mask = pad_batch(all_token_weights, max_len, np.float32)
        sampling_logprobs = pad_batch(all_sampling_logprobs, max_len, np.float32)
        advantages = pad_batch(all_advantages, max_len, np.float32)

        total_bs = int(input_ids.shape[0])
        micro_bs = self._micro_batch_size(total_bs)
        seq_lens = [len(seq) for seq in all_input_ids]
        seq_len = input_ids.shape[1]

        token_losses_device = []
        logprobs_device = []

        with jax.set_mesh(self.mesh), self._jit_timing_context(seq_len, mode="train"):
            for mb_start in range(0, total_bs, micro_bs):
                mb_end = min(mb_start + micro_bs, total_bs)
                self.accumulated_grads, per_token_losses, target_logprobs, _ = self._forward_backward_and_accumulate(
                    self.accumulated_grads,
                    self.model,
                    input_ids[mb_start:mb_end],
                    attention_mask[mb_start:mb_end],
                    target_ids[mb_start:mb_end],
                    loss_mask[mb_start:mb_end],
                    loss_fn_types[mb_start:mb_end],
                    sampling_logprobs[mb_start:mb_end],
                    advantages[mb_start:mb_end],
                )
                token_losses_device.append(per_token_losses)
                logprobs_device.append(target_logprobs)

        token_losses_host, logprobs_host = jax.device_get((token_losses_device, logprobs_device))

        token_losses_out = []
        logprobs_out = []
        idx = 0
        for mb_losses, mb_logprobs in zip(token_losses_host, logprobs_host):
            for i in range(mb_losses.shape[0]):
                token_losses_out.append(mb_losses[i, :seq_lens[idx]].astype(jnp.float32))
                logprobs_out.append(mb_logprobs[i, :seq_lens[idx]].astype(jnp.float32))
                idx += 1

        for request_id, _, start_idx, end_idx in request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                token_losses = token_losses_out[i]
                token_logprobs = logprobs_out[i]
                loss_fn_outputs.append({
                    "elementwise_loss": {
                        "data": token_losses.tolist(),
                        "dtype": "float32",
                        "shape": [token_losses.shape[0]],
                    },
                    "logprobs": {
                        "data": token_logprobs.tolist(),
                        "dtype": "float32",
                        "shape": [token_logprobs.shape[0]],
                    },
                })
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )

        return results

    def process_optim_step(
        self,
        model_id: str,
        request_data: types.OptimStepInput,
        optimizer: nnx.Optimizer,
        adapter_index: int | None = None,
    ) -> types.OptimStepOutput:
        """Process an optimizer step request."""
        if self.accumulated_grads.count[0] == 0:
            logger.warning(f"No accumulated gradients for model {model_id}, skipping optimizer step")
            return types.OptimStepOutput()

        hp = optimizer.opt_state.hyperparams
        hp["learning_rate"][...] = request_data.adam_params.learning_rate
        hp["b1"][...] = request_data.adam_params.beta1
        hp["b2"][...] = request_data.adam_params.beta2
        hp["eps"][...] = request_data.adam_params.eps

        adapter_index_arr = jnp.int32(adapter_index) if adapter_index is not None else jnp.int32(0)

        with jax.set_mesh(self.mesh):
            self.accumulated_grads = self._compute_grads_and_update(
                self.accumulated_grads,
                self.lora_params,
                optimizer,
                adapter_index_arr,
            )

        logger.info(f"Applied optimizer step for model {model_id} (adapter {adapter_index})")
        return types.OptimStepOutput()

    def process_sample_batch(
        self,
        requests: dict[str, tuple[str, types.SampleInput]],
        models: dict[str, types.ModelMetadata],
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process multiple sample requests in a single batch."""
        if not requests:
            return {}

        results = {}
        needs_prompt_logprobs = any(request_data.prompt_logprobs for (_, request_data) in requests.values())

        all_prompts = []
        all_sampling_params = []
        all_adapter_indices = []
        request_batch_slices = []

        # Load sampler weights and get adapter indices
        adapter_indices_batch = []
        for model_id, request_data in requests.values():
            if request_data.base_model is None:
                adapter_indices_batch.append(models[model_id].adapter_index)
            else:
                adapter_indices_batch.append(0)

        for i, (request_id, (model_id, request_data)) in enumerate(requests.items()):
            request_start = len(all_prompts)
            if model_id and model_id not in models:
                logger.warning(f"Model {model_id} not loaded, skipping sample request")
                results[request_id] = types.ErrorResponse(error=f"Model {model_id} not loaded", status="failed")
                continue

            for _ in range(request_data.num_samples):
                prompt_tokens = [token for chunk in request_data.prompt.chunks for token in chunk.tokens]
                all_prompts.append(prompt_tokens)
                all_sampling_params.append(request_data.sampling_params)
                all_adapter_indices.append(adapter_indices_batch[i])

            request_batch_slices.append((request_id, model_id, request_start, len(all_prompts), request_data))

        if not all_prompts:
            return results

        total_batch_size = len(all_prompts)
        max_batch_size = (
            self.config.sample_max_num_sequences if self.config.sample_max_num_sequences > 0 else total_batch_size
        )

        all_sequences: list[types.GeneratedSequence] = []
        all_prompt_logprobs: list[list[float]] = []

        with jax.set_mesh(self.mesh):
            model = nnx.merge(self.graphdef, self.lora_params, self.non_lora_params)
            for batch_start in range(0, total_batch_size, max_batch_size):
                batch_end = min(batch_start + max_batch_size, total_batch_size)
                batch_prompts = pad(all_prompts[batch_start:batch_end], max_batch_size, fill=[])
                adapter_indices = pad(all_adapter_indices[batch_start:batch_end], max_batch_size, fill=0)
                sampling_params = pad(
                    all_sampling_params[batch_start:batch_end], max_batch_size, fill=all_sampling_params[batch_start]
                )

                max_len = round_up_seq_len(max((len(seq) for seq in batch_prompts), default=0), self.config.min_seq_len)
                input_ids = pad_batch(batch_prompts, max_len, np.int32, left=True)
                attention_mask = pad_batch([[1] * len(seq) for seq in batch_prompts], max_len, np.int32, left=True)

                with self._jit_timing_context(max_len, mode="sample"):
                    result = model.generate(
                        input_ids,
                        attention_mask,
                        sampling_params=sampling_params,
                        adapter_indices=jnp.array(adapter_indices, dtype=jnp.int32),
                        prompt_logprobs=needs_prompt_logprobs,
                    )

                batch_size = batch_end - batch_start
                all_sequences.extend(
                    types.GeneratedSequence(stop_reason=stop_reason, tokens=tokens, logprobs=logprobs)
                    for stop_reason, tokens, logprobs in zip(
                        result.stop_reasons[:batch_size],
                        result.generated_ids[:batch_size],
                        result.logprobs[:batch_size],
                    )
                )
                if needs_prompt_logprobs and result.prompt_logprobs:
                    all_prompt_logprobs.extend(result.prompt_logprobs[:batch_size])

        for request_id, _, start_idx, end_idx, request_data in request_batch_slices:
            sequences = [all_sequences[i] for i in range(start_idx, end_idx)]
            prompt_logprobs = (
                all_prompt_logprobs[start_idx] if request_data.prompt_logprobs and all_prompt_logprobs else None
            )
            results[request_id] = types.SampleOutput(sequences=sequences, prompt_logprobs=prompt_logprobs)

        return results

    def save_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
        optimizers: dict[str, nnx.Optimizer],
    ) -> None:
        """Save training checkpoint as tar.gz using Flax checkpoints."""
        with pack_and_upload(output_path) as temp_dir:
            checkpoint_data = self.extract_checkpoint_data(model_id, models, optimizers)
            checkpoints.save_checkpoint(
                target=checkpoint_data,
                ckpt_dir=temp_dir,
                step=0,
                prefix="checkpoint_",
                overwrite=True,
            )
        logger.info(f"Saved training checkpoint to {output_path}")

    def extract_checkpoint_data(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
        optimizers: dict[str, nnx.Optimizer],
    ) -> dict:
        """Extract adapter state and optimizer state for checkpointing."""
        adapter_index = models[model_id].adapter_index
        rank = models[model_id].lora_config.rank
        lora_weights = extract_adapter_state(adapter_index, self.lora_params, rank)
        optimizer_state = extract_adapter_state(adapter_index, nnx.state(optimizers[model_id]), rank)
        return {
            "lora_weights": lora_weights,
            "optimizer_state": optimizer_state,
            "lora_config": models[model_id].lora_config.model_dump(),
        }

    def insert_checkpoint_data(
        self,
        model_id: str,
        checkpoint_data: dict,
        models: dict[str, types.ModelMetadata],
        optimizers: dict[str, nnx.Optimizer],
    ) -> None:
        """Insert checkpoint data into model state."""
        adapter_index = models[model_id].adapter_index
        rank = checkpoint_data["lora_config"]["rank"]

        if models[model_id].lora_config.rank != rank:
            raise ValueError(
                f"Rank mismatch: checkpoint has rank {rank}, model configured with rank {models[model_id].lora_config.rank}"
            )

        insert_adapter_state(adapter_index, self.lora_params, checkpoint_data["lora_weights"], rank)
        insert_adapter_state(adapter_index, nnx.state(optimizers[model_id]), checkpoint_data["optimizer_state"], rank)

    def save_sampler_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save sampler checkpoint as tar.gz using save_lora_checkpoint."""
        save_lora_checkpoint(
            self.model,
            self.config.base_model,
            models[model_id].lora_config,
            models[model_id].adapter_index,
            output_path,
        )
        logger.info(f"Saved LoRA sampler checkpoint to {output_path}")

    def extract_sampler_weights(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract weights for sampler checkpoint - returns data needed for save_lora_checkpoint."""
        return {
            "model": self.model,
            "base_model": self.config.base_model,
            "lora_config": models[model_id].lora_config,
            "adapter_index": models[model_id].adapter_index,
        }

    def insert_sampler_weights(
        self,
        model_id: str,
        checkpoint_id: str,
        checkpoint_path,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert sampler weights from checkpoint file."""
        adapter_index = models[model_id].adapter_index
        adapter_config = models[model_id].lora_config
        load_lora_checkpoint(self.model, adapter_config, adapter_index, checkpoint_path)
        models[model_id].loaded_checkpoint_id = checkpoint_id
        logger.info(f"Loaded LoRA sampler weights for model {model_id} at adapter index {adapter_index}")

"""MaxText backend for TinkerEngine."""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import checkpoints

from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.backends.backend import AbstractBackend
from tx.tinker.backends.utils import pad_batch
from tx.tinker.loss_fns import LOSS_FUNCTIONS, LOSS_TYPES
from tx.utils.models import round_up_seq_len, convert_maxtext_lora_to_hf
from tx.utils.storage import pack_and_upload
from observability import log, bootstrap, set_run_id, Events

# MaxText imports
import MaxText
from MaxText import maxtext_utils
from MaxText import model_creation_utils as maxtext_model_creation
from MaxText import sharding as maxtext_sharding
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter



def compute_token_mis_metrics(
    target_logprobs: jax.Array,
    sampling_logprobs: jax.Array,
    loss_mask: jax.Array,
    advantages: jax.Array,
    clip_low: jax.Array,
    clip_high: jax.Array,
) -> dict[str, float]:
    """Compute token-level MIS diagnostic metrics.

    Computes metrics about importance ratio distribution and mask activation patterns.
    Only uses masked (valid) tokens for statistics.

    Args:
        target_logprobs: [batch, seq] log probs from learner policy
        sampling_logprobs: [batch, seq] log probs from sampling policy
        loss_mask: [batch, seq] mask for valid tokens
        advantages: [batch, seq] per-token advantages
        clip_low: [batch] lower clipping threshold per sequence
        clip_high: [batch] upper clipping threshold per sequence

    Returns:
        Dict of metrics to log to wandb.
    """
    # Per-token log ratio and importance ratio
    log_ratio = target_logprobs - sampling_logprobs
    prob_ratio = jnp.exp(log_ratio)

    # Flatten and select only valid tokens
    flat_log_ratio = log_ratio.flatten()
    flat_prob_ratio = prob_ratio.flatten()
    flat_mask = loss_mask.flatten()
    flat_advantages = advantages.flatten()

    valid_mask = flat_mask > 0
    valid_log_ratio = jnp.where(valid_mask, flat_log_ratio, jnp.nan)
    valid_prob_ratio = jnp.where(valid_mask, flat_prob_ratio, jnp.nan)
    valid_advantages = jnp.where(valid_mask, flat_advantages, jnp.nan)

    n_valid = valid_mask.sum()

    # Broadcast clip thresholds to token level: [batch, 1] -> [batch, seq]
    clip_low_expanded = jnp.broadcast_to(clip_low[:, None], log_ratio.shape)
    clip_high_expanded = jnp.broadcast_to(clip_high[:, None], log_ratio.shape)
    flat_clip_low = clip_low_expanded.flatten()
    flat_clip_high = clip_high_expanded.flatten()

    # Token-level mask: which tokens would be zeroed by token_mis
    in_bounds = (prob_ratio >= clip_low_expanded) & (prob_ratio <= clip_high_expanded)
    flat_in_bounds = in_bounds.flatten()
    token_mis_mask = jnp.where(valid_mask, flat_in_bounds.astype(jnp.float32), jnp.nan)

    # Mask activation rates
    masked_low = jnp.where(valid_mask, (flat_prob_ratio < flat_clip_low).astype(jnp.float32), 0.0)
    masked_high = jnp.where(valid_mask, (flat_prob_ratio > flat_clip_high).astype(jnp.float32), 0.0)

    mask_rate_total = 1.0 - jnp.nanmean(token_mis_mask)
    mask_rate_low = masked_low.sum() / n_valid
    mask_rate_high = masked_high.sum() / n_valid

    # Mask rate by advantage sign
    pos_adv_mask = valid_mask & (flat_advantages > 0)
    neg_adv_mask = valid_mask & (flat_advantages < 0)
    n_pos = pos_adv_mask.sum()
    n_neg = neg_adv_mask.sum()

    masked_given_pos_adv = jnp.where(pos_adv_mask, (~flat_in_bounds).astype(jnp.float32), 0.0).sum()
    masked_given_neg_adv = jnp.where(neg_adv_mask, (~flat_in_bounds).astype(jnp.float32), 0.0).sum()
    mask_rate_pos_adv = jnp.where(n_pos > 0, masked_given_pos_adv / n_pos, 0.0)
    mask_rate_neg_adv = jnp.where(n_neg > 0, masked_given_neg_adv / n_neg, 0.0)

    # Effective gradient mass: ratio of masked loss to total loss
    unmasked_loss_mass = jnp.where(valid_mask & flat_in_bounds, flat_prob_ratio * jnp.abs(flat_advantages), 0.0).sum()
    total_loss_mass = jnp.where(valid_mask, flat_prob_ratio * jnp.abs(flat_advantages), 0.0).sum()
    effective_gradient_frac = jnp.where(total_loss_mass > 0, unmasked_loss_mass / total_loss_mass, 1.0)

    # Percentiles via sorting (JAX doesn't have jnp.percentile, so we use sorted indexing)
    # For efficiency, compute a few key percentiles
    sorted_ratio = jnp.sort(jnp.where(valid_mask, flat_prob_ratio, jnp.inf))
    n_valid_int = int(n_valid)

    def safe_percentile(sorted_arr, p, n):
        idx = jnp.clip(jnp.floor(p * n / 100).astype(jnp.int32), 0, n - 1)
        return sorted_arr[idx]

    # Metric names include reduction type suffix (required by tinker SDK's _metrics_reduction)
    # :mean = weighted average by loss_fn_outputs count when combining chunked requests
    # :sum = sum across chunks
    return {
        "mis/log_ratio_mean:mean": float(jnp.nanmean(valid_log_ratio)),
        "mis/log_ratio_std:mean": float(jnp.nanstd(valid_log_ratio)),
        "mis/prob_ratio_mean:mean": float(jnp.nanmean(valid_prob_ratio)),
        "mis/prob_ratio_std:mean": float(jnp.nanstd(valid_prob_ratio)),
        "mis/prob_ratio_p5:mean": float(safe_percentile(sorted_ratio, 5, n_valid)),
        "mis/prob_ratio_p50:mean": float(safe_percentile(sorted_ratio, 50, n_valid)),
        "mis/prob_ratio_p95:mean": float(safe_percentile(sorted_ratio, 95, n_valid)),
        "mis/mask_rate_total:mean": float(mask_rate_total),
        "mis/mask_rate_low:mean": float(mask_rate_low),
        "mis/mask_rate_high:mean": float(mask_rate_high),
        "mis/mask_rate_pos_adv:mean": float(mask_rate_pos_adv),
        "mis/mask_rate_neg_adv:mean": float(mask_rate_neg_adv),
        "mis/effective_gradient_frac:mean": float(effective_gradient_frac),
        "mis/n_valid_tokens:sum": int(n_valid),
    }


def reset_adapter_weights(model):

    state = nnx.state(model)

    def update_lora_config(path, value):
        normalized_path = tuple(p.key if hasattr(p, "key") else p.name for p in path)
        j_path  = "/".join(normalized_path)
        if "lora_b" in j_path:
            return value.at[...].set(0.0)
        return value

    updated_state = jax.tree.map_with_path(update_lora_config, state)
    nnx.update(model, updated_state)

def _get_maxtext_base_config_path() -> str:
    """Get the absolute path to MaxText's base.yml config file."""
    import os
    from importlib.resources import files
    try:
        # Try importlib.resources first (works if MaxText packages configs properly)
        config_path = str(files("MaxText").joinpath("configs", "base.yml"))
        if os.path.exists(config_path):
            return config_path
    except (TypeError, FileNotFoundError):
        pass
    # Fallback: derive from package location
    maxtext_pkg_dir = os.path.dirname(MaxText.__file__)
    maxtext_root = os.path.dirname(os.path.dirname(maxtext_pkg_dir))
    config_path = os.path.join(maxtext_root, "src", "MaxText", "configs", "base.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find MaxText base.yml config. Tried: {config_path}. "
            "Ensure MaxText is installed correctly or set MAXTEXT_CONFIG_PATH environment variable."
        )
    return config_path


def parse_maxtext_config(config_str: str):
    """Parse MaxText config from space-separated key=value string."""
    if not config_str:
        return None
    config_path = _get_maxtext_base_config_path()
    log.info("using maxtext config", component="maxtext", config_path=config_path)
    argv = ["", config_path] + config_str.split()
    from MaxText import pyconfig as maxtext_pyconfig
    return maxtext_pyconfig.initialize(argv)



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


class MaxTextBackend(AbstractBackend):
    """Backend for MaxText models with context parallelism.

    This is a single-adapter backend (max_lora_adapters must be 1).
    """

    def __init__(self, config: EngineConfig, maxtext_config):
        """Initialize MaxText backend."""
        if config.max_lora_adapters != 1:
            raise ValueError(
                f"MaxTextBackend only supports single adapter (max_lora_adapters=1), "
                f"got max_lora_adapters={config.max_lora_adapters}"
            )

        self.config = config
        self.maxtext_config = maxtext_config
        self.metrics = types.EngineMetrics()

        # Create mesh using MaxText's device mesh creation
        devices_array = maxtext_utils.create_device_mesh(maxtext_config)
        self.mesh = jax.sharding.Mesh(devices_array, maxtext_config.mesh_axes)
        log.info("created mesh", component="maxtext", shape=str(self.mesh.shape), axes=str(self.mesh.axis_names))

        # Create model using MaxText's model creation
        with jax.set_mesh(self.mesh):
            base_model, _ = maxtext_model_creation.create_nnx_model(maxtext_config, mesh=self.mesh)
            self.model = TunixMaxTextAdapter(base_model=base_model)
            self.model.config = None

        # Extract LoRA params for gradient accumulation
        lora_filter = nnx.All(nnx.Param, nnx.Any(nnx.PathContains("lora_a"), nnx.PathContains("lora_b")))
        self.lora_filter = lora_filter
        self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, lora_filter, ...)

        # Initialize accumulated gradients
        self.accumulated_grads = AccumulatedGradients.create(self.lora_params)
        self._log_accumulated_grads()

        # Per-model optimizer storage (managed internally)
        self.optimizers: dict[str, nnx.Optimizer] = {}

        log.info("initialized model", component="maxtext", context_parallel_size=maxtext_config.context_parallel_size)

        self._create_loss_and_grad_fn()

    def _log_accumulated_grads(self):
        """Log accumulated gradient structure."""
        accum_params = _count_params(self.accumulated_grads.grad_sum)
        log.info("accumulated grads total params", component="maxtext", params_millions=round(accum_params / 1e6, 2))
        for path, val in jax.tree_util.tree_leaves_with_path(self.accumulated_grads.grad_sum):
            path_str = "/".join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
            log.info("accumulated grad shape", component="maxtext", path=path_str, shape=str(val.shape))

    def _create_loss_and_grad_fn(self):
        """Create loss and gradient functions for MaxText model."""

        # TODO: must fix this slop - clip_low/clip_high are parallel arrays, should be bundled with other per-example data
        def loss_for_maxtext_model(
            model,
            input_ids: jax.Array,
            positions: jax.Array,
            target_ids: jax.Array,
            loss_mask: jax.Array,
            loss_fn_types: jax.Array,
            sampling_logprobs: jax.Array,
            advantages: jax.Array,
            clip_low: jax.Array,
            clip_high: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            """Loss for MaxText model with dynamic loss function selection."""
            logits, _ = model(input_ids, positions, None, None, False)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)

            # TODO: must fix this slop - clip_low/clip_high passed separately instead of bundled config
            def compute_loss_per_example(loss_fn_type, target_logprobs, loss_mask, sampling_logprobs, advantages, clip_low, clip_high):
                return jax.lax.switch(
                    loss_fn_type,
                    LOSS_FUNCTIONS,
                    target_logprobs,
                    loss_mask,
                    sampling_logprobs,
                    advantages,
                    clip_low,
                    clip_high,
                )

            per_token_losses = jax.vmap(compute_loss_per_example)(
                loss_fn_types,
                target_logprobs,
                loss_mask,
                sampling_logprobs,
                advantages,
                clip_low,
                clip_high,
            )

            per_seq_loss = per_token_losses.sum(axis=-1) / jnp.maximum(loss_mask.sum(axis=-1), 1e-9)
            total_loss = per_seq_loss.sum()
            return total_loss, (target_logprobs, per_token_losses)

        loss_and_grad_fn = nnx.value_and_grad(
            loss_for_maxtext_model,
            argnums=nnx.DiffState(0, self.lora_filter),
            has_aux=True
        )

        def forward_backward_maxtext(
            model, input_ids, positions, target_ids, loss_mask, loss_fn_types, sampling_logprobs, advantages, clip_low, clip_high,
        ) -> tuple[jax.Array, jax.Array, jax.Array, nnx.State]:
            """Forward-backward for MaxText model."""
            (loss, (target_logprobs, per_token_losses)), grads = loss_and_grad_fn(
                model, input_ids, positions, target_ids, loss_mask, loss_fn_types, sampling_logprobs, advantages, clip_low, clip_high,
            )
            return loss, target_logprobs, per_token_losses, grads

        data_sharding = maxtext_sharding.get_input_data_sharding(self.maxtext_config, self.mesh)

        if self.config.enforce_eager:
            self._forward_backward = forward_backward_maxtext
        else:
            with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
                self._forward_backward = jax.jit(
                    forward_backward_maxtext,
                    # model, input_ids, positions, target_ids, loss_mask, loss_fn_types (1D), sampling_logprobs, advantages, clip_low (1D), clip_high (1D)
                    in_shardings=(None, data_sharding, data_sharding, data_sharding, data_sharding, None, data_sharding, data_sharding, None, None),
                )

        def optim_step(model, optimizer, grads):
            """Apply gradients to optimizer."""
            optimizer.update(model, grads)

        if self.config.enforce_eager:
            self._optim_step = optim_step
        else:
            self._optim_step = nnx.jit(optim_step)

        log.info("created loss and gradient functions", component="maxtext")

    def _micro_batch_size(self, total: int) -> int:
        """Return effective micro-batch size."""
        mb = self.config.train_micro_batch_size
        return total if mb <= 0 else max(1, min(mb, total))

    @contextmanager
    def _jit_timing_context(self, seq_len: int, mode: str):
        """Context manager to track JIT compilation times."""
        jit_times = self.metrics.train_seq_len_jit_times if mode == "train" else self.metrics.sample_seq_len_jit_times
        if not self.config.enforce_eager and seq_len not in jit_times:
            log.info("jit compiling in progress", component="maxtext", mode=mode, seq_len=seq_len)
            start_time = time.time()
            yield
            elapsed = time.time() - start_time
            jit_times[seq_len] = elapsed
            log.info("jit compilation complete", component="maxtext", mode=mode, seq_len=seq_len, elapsed_s=round(elapsed, 2))
        else:
            yield

    def register_model(self, model_id: str, adapter_index: int, lora_config: types.LoraConfig) -> None:
        """Register a new model with the backend.

        Creates optimizer for the model. MaxText is single-adapter so adapter_index is ignored.
        """
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
        with jax.set_mesh(self.mesh):
            self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)
        log.info("registered model", component="maxtext", model_id=model_id)

    def unregister_model(self, model_id: str, adapter_index: int) -> None:
        """Unregister a model from the backend.

        Removes optimizer and resets LoRA weights (only zeros lora_b, preserves lora_a).
        """
        self.optimizers.pop(model_id, None)

        # Reset LoRA weights (only zero lora_b, preserve lora_a random init for gradients to flow)
        reset_adapter_weights(self.model)
        # Re-split to update lora_params reference
        self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, self.lora_filter, ...)
        log.info("unregistered model", component="maxtext", model_id=model_id)

    def precompile_kernels(self, seq_lens: list[int]):
        """Precompile JIT kernels for specified sequence lengths."""
        if not seq_lens or self.config.enforce_eager:
            return

        log.info("precompiling jit kernels", component="maxtext", seq_lens=seq_lens)
        micro_bs = max(1, self.config.train_micro_batch_size) if self.config.train_micro_batch_size > 0 else 1

        with jax.set_mesh(self.mesh):
            for seq_len in seq_lens:
                dummy_input_ids = jnp.zeros((micro_bs, seq_len), dtype=jnp.int32)
                dummy_target_ids = jnp.zeros((micro_bs, seq_len), dtype=jnp.int32)
                dummy_loss_mask = jnp.ones((micro_bs, seq_len), dtype=jnp.float32)
                dummy_positions = jnp.broadcast_to(jnp.arange(seq_len), (micro_bs, seq_len))
                dummy_loss_fn_types = jnp.zeros((micro_bs,), dtype=jnp.int32)
                dummy_sampling_logprobs = jnp.zeros((micro_bs, seq_len), dtype=jnp.float32)
                dummy_advantages = jnp.zeros((micro_bs, seq_len), dtype=jnp.float32)

                data_sharding = maxtext_sharding.get_input_data_sharding(self.maxtext_config, self.mesh)
                dummy_input_ids = jax.device_put(dummy_input_ids, data_sharding)
                dummy_positions = jax.device_put(dummy_positions, data_sharding)
                dummy_target_ids = jax.device_put(dummy_target_ids, data_sharding)
                dummy_loss_mask = jax.device_put(dummy_loss_mask, data_sharding)
                # dummy_loss_fn_types is 1D, no sharding needed
                dummy_sampling_logprobs = jax.device_put(dummy_sampling_logprobs, data_sharding)
                dummy_advantages = jax.device_put(dummy_advantages, data_sharding)

                # TODO: must fix this slop - hardcoded default clip values
                dummy_clip_low = jnp.full((micro_bs,), 0.8, dtype=jnp.float32)
                dummy_clip_high = jnp.full((micro_bs,), 1.2, dtype=jnp.float32)

                with nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
                    with self._jit_timing_context(seq_len, mode="train"):
                        _, _, _, grads = self._forward_backward(
                            self.model, dummy_input_ids, dummy_positions, dummy_target_ids, dummy_loss_mask,
                            dummy_loss_fn_types, dummy_sampling_logprobs, dummy_advantages,
                            dummy_clip_low, dummy_clip_high,
                        )
                        self.accumulated_grads = self.accumulated_grads.add(grads, micro_bs)

                self.accumulated_grads = AccumulatedGradients.create(self.lora_params)

        log.info("precompilation complete", component="maxtext", num_seq_lens=len(seq_lens))

    def _process_forward_backward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> tuple[list[dict], dict[str, float]]:
        """Process forward_backward requests using MaxText model (single bucket).

        Returns:
            Tuple of:
            - List of per-sequence outputs, where each dict contains 'elementwise_loss' and 'logprobs'.
              The list is indexed by position in the bucket (not by request_id).
            - Dict of batch-level metrics (e.g., MIS diagnostics)
        """
        all_input_ids = prepared_batch.all_input_ids
        all_targets = prepared_batch.all_targets
        all_token_weights = prepared_batch.all_token_weights
        all_sampling_logprobs = prepared_batch.all_sampling_logprobs
        all_advantages = prepared_batch.all_advantages
        all_loss_fn_types = prepared_batch.all_loss_fn_types
        all_clip_low = prepared_batch.all_clip_low  # TODO: must fix this slop - parallel arrays
        all_clip_high = prepared_batch.all_clip_high  # TODO: must fix this slop - parallel arrays

        if not all_input_ids:
            return [], {}

        max_len = round_up_seq_len(max(len(seq) for seq in all_input_ids), self.config.min_seq_len)
        input_ids = pad_batch(all_input_ids, max_len, np.int32)
        target_ids = pad_batch(all_targets, max_len, np.int32)
        loss_mask = pad_batch(all_token_weights, max_len, np.float32)
        sampling_logprobs = pad_batch(all_sampling_logprobs, max_len, np.float32)
        advantages = pad_batch(all_advantages, max_len, np.float32)
        loss_fn_types = jnp.array(all_loss_fn_types, dtype=jnp.int32)
        clip_low = jnp.array(all_clip_low, dtype=jnp.float32)  # TODO: must fix this slop - parallel arrays
        clip_high = jnp.array(all_clip_high, dtype=jnp.float32)  # TODO: must fix this slop - parallel arrays

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Pad batch to be divisible by FSDP size for sharding compatibility
        fsdp_size = self.mesh.shape['fsdp']
        original_batch_size = batch_size
        if batch_size % fsdp_size != 0:
            padded_batch_size = ((batch_size + fsdp_size - 1) // fsdp_size) * fsdp_size
            pad_size = padded_batch_size - batch_size
            input_ids = np.pad(input_ids, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            target_ids = np.pad(target_ids, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            loss_mask = np.pad(loss_mask, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            sampling_logprobs = np.pad(sampling_logprobs, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            advantages = np.pad(advantages, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            loss_fn_types = jnp.pad(loss_fn_types, (0, pad_size), mode='constant', constant_values=0)
            # TODO: must fix this slop - parallel arrays need manual padding
            clip_low = jnp.pad(clip_low, (0, pad_size), mode='constant', constant_values=0.8)
            clip_high = jnp.pad(clip_high, (0, pad_size), mode='constant', constant_values=1.2)
            batch_size = padded_batch_size
            log.info("padded batch for fsdp sharding", component="maxtext", original_batch_size=original_batch_size, padded_batch_size=padded_batch_size)

        positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
        seq_lens = [len(seq) for seq in all_input_ids]

        data_sharding = maxtext_sharding.get_input_data_sharding(self.maxtext_config, self.mesh)

        token_losses_device = []
        logprobs_device = []
        total_bs = batch_size
        micro_bs = self._micro_batch_size(total_bs)

        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
            with self._jit_timing_context(seq_len, mode="train"):
                for mb_start in range(0, total_bs, micro_bs):
                    mb_end = min(mb_start + micro_bs, total_bs)
                    log.info("forward-backward batch", component="maxtext", mb_start=mb_start, mb_end=mb_end, seq_len=seq_len)
                    tic = time.time()

                    mb_input_ids = jax.device_put(input_ids[mb_start:mb_end], data_sharding)
                    mb_positions = jax.device_put(positions[mb_start:mb_end], data_sharding)
                    mb_target_ids = jax.device_put(target_ids[mb_start:mb_end], data_sharding)
                    mb_loss_mask = jax.device_put(loss_mask[mb_start:mb_end], data_sharding)
                    mb_loss_fn_types = loss_fn_types[mb_start:mb_end]  # 1D, no sharding needed
                    mb_sampling_logprobs = jax.device_put(sampling_logprobs[mb_start:mb_end], data_sharding)
                    mb_advantages = jax.device_put(advantages[mb_start:mb_end], data_sharding)
                    mb_clip_low = clip_low[mb_start:mb_end]  # 1D, no sharding needed
                    mb_clip_high = clip_high[mb_start:mb_end]  # 1D, no sharding needed

                    _, target_logprobs, per_token_losses, grads = self._forward_backward(
                        self.model,
                        mb_input_ids,
                        mb_positions,
                        mb_target_ids,
                        mb_loss_mask,
                        mb_loss_fn_types,
                        mb_sampling_logprobs,
                        mb_advantages,
                        mb_clip_low,
                        mb_clip_high,
                    )

                    _ = jax.device_get(target_logprobs)

                    took = time.time() - tic
                    tokens_processed = (mb_end - mb_start) * seq_len
                    tokens_per_sec = tokens_processed / took if took > 0 else float('nan')
                    log.info("forward-backward complete", component="maxtext", mb_start=mb_start, mb_end=mb_end, elapsed_s=round(took, 3), tokens_per_sec=round(tokens_per_sec, 1))

                    micro_batch_size = mb_end - mb_start
                    self.accumulated_grads = self.accumulated_grads.add(grads, micro_batch_size)
                    token_losses_device.append(per_token_losses)
                    logprobs_device.append(target_logprobs)

        token_losses_host, logprobs_host = jax.device_get((token_losses_device, logprobs_device))

        token_losses_out = []
        logprobs_out = []
        idx = 0
        for mb_losses, mb_logprobs in zip(token_losses_host, logprobs_host):
            for i in range(mb_losses.shape[0]):
                if idx >= original_batch_size:
                    break  # Skip padded samples
                token_losses_out.append(mb_losses[i, :seq_lens[idx]].astype(jnp.float32))
                logprobs_out.append(mb_logprobs[i, :seq_lens[idx]].astype(jnp.float32))
                idx += 1
            if idx >= original_batch_size:
                break

        # Compute MIS metrics if using token_mis or seq_mis loss
        # Metrics are computed on the full (pre-truncation) padded arrays for efficiency
        batch_metrics = {}
        mis_loss_types = {LOSS_TYPES.get("token_mis"), LOSS_TYPES.get("seq_mis")}
        if any(lt in mis_loss_types for lt in all_loss_fn_types):
            # Concatenate all micro-batch logprobs to compute metrics on full batch
            all_target_logprobs = jnp.concatenate(logprobs_device, axis=0)[:original_batch_size]
            batch_metrics = compute_token_mis_metrics(
                target_logprobs=all_target_logprobs,
                sampling_logprobs=jnp.array(sampling_logprobs[:original_batch_size]),
                loss_mask=jnp.array(loss_mask[:original_batch_size]),
                advantages=jnp.array(advantages[:original_batch_size]),
                clip_low=clip_low[:original_batch_size],
                clip_high=clip_high[:original_batch_size],
            )

        # Return per-sequence outputs (indexed by bucket position)
        per_seq_outputs = []
        for i in range(len(token_losses_out)):
            token_losses = token_losses_out[i]
            token_logprobs = logprobs_out[i]
            per_seq_outputs.append({
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

        return per_seq_outputs, batch_metrics

    def split_batch_by_bucket(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[int, tuple[types.PreparedModelPassBatch, list[int]]]:
        """Split a prepared batch into buckets based on rounded-up sequence length.

        Uses round_up_seq_len to bucket sequences, so sequences of similar lengths
        are processed together, minimizing padding overhead.

        Args:
            prepared_batch: The full batch to split

        Returns:
            Dict mapping bucket_seq_len -> (PreparedModelPassBatch, seq_idx_map)
            where seq_idx_map[bucket_idx] = original_seq_idx
        """
        buckets: dict[int, dict] = {}

        for seq_idx, input_ids in enumerate(prepared_batch.all_input_ids):
            bucket_key = round_up_seq_len(len(input_ids), self.config.min_seq_len)

            if bucket_key not in buckets:
                buckets[bucket_key] = {
                    "all_input_ids": [],
                    "all_targets": [],
                    "all_token_weights": [],
                    "all_sampling_logprobs": [],
                    "all_advantages": [],
                    "all_adapter_indices": [],
                    "all_loss_fn_types": [],
                    "all_clip_low": [],  # TODO: must fix this slop - parallel arrays
                    "all_clip_high": [],  # TODO: must fix this slop - parallel arrays
                    "seq_idx_map": [],  # Maps bucket index -> original index
                }

            bucket = buckets[bucket_key]
            bucket["all_input_ids"].append(input_ids)
            bucket["all_targets"].append(prepared_batch.all_targets[seq_idx])
            bucket["all_token_weights"].append(prepared_batch.all_token_weights[seq_idx])
            bucket["all_sampling_logprobs"].append(prepared_batch.all_sampling_logprobs[seq_idx])
            bucket["all_advantages"].append(prepared_batch.all_advantages[seq_idx])
            bucket["all_adapter_indices"].append(prepared_batch.all_adapter_indices[seq_idx])
            bucket["all_loss_fn_types"].append(prepared_batch.all_loss_fn_types[seq_idx])
            bucket["all_clip_low"].append(prepared_batch.all_clip_low[seq_idx])
            bucket["all_clip_high"].append(prepared_batch.all_clip_high[seq_idx])
            bucket["seq_idx_map"].append(seq_idx)

        result = {}
        for bucket_key, bucket_data in buckets.items():
            seq_idx_map = bucket_data.pop("seq_idx_map")

            # request_batch_slices not needed - aggregation happens in process_forward_backward_batch
            batch = types.PreparedModelPassBatch(
                all_input_ids=bucket_data["all_input_ids"],
                all_targets=bucket_data["all_targets"],
                all_token_weights=bucket_data["all_token_weights"],
                all_sampling_logprobs=bucket_data["all_sampling_logprobs"],
                all_advantages=bucket_data["all_advantages"],
                all_adapter_indices=bucket_data["all_adapter_indices"],
                all_loss_fn_types=bucket_data["all_loss_fn_types"],
                all_clip_low=bucket_data["all_clip_low"],
                all_clip_high=bucket_data["all_clip_high"],
                request_batch_slices=[],
            )
            result[bucket_key] = (batch, seq_idx_map)

        return result

    def process_forward_backward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward_backward requests, bucketing by sequence length to minimize padding.

        Splits the batch into buckets based on rounded-up sequence length, processes
        each bucket separately, and merges results while preserving original sequence order.
        """
        if not prepared_batch.all_input_ids:
            return {}

        buckets = self.split_batch_by_bucket(prepared_batch)

        if len(buckets) > 1:
            bucket_sizes = {k: len(v[0].all_input_ids) for k, v in buckets.items()}
            log.info("split batch into buckets", component="maxtext", num_buckets=len(buckets), bucket_sizes=bucket_sizes)

        # Collect per-sequence outputs indexed by original sequence index
        per_seq_outputs: dict[int, dict] = {}
        # Aggregate metrics across all buckets (use last bucket's metrics as they're batch-level)
        aggregated_metrics: dict[str, float] = {}

        for bucket_seq_len, (bucket_batch, seq_idx_map) in sorted(buckets.items()):
            bucket_outputs, bucket_metrics = self._process_forward_backward_batch(bucket_batch)

            # Map bucket outputs back to original sequence indices
            for bucket_idx, output in enumerate(bucket_outputs):
                orig_idx = seq_idx_map[bucket_idx]
                per_seq_outputs[orig_idx] = output

            # Merge bucket metrics (later buckets overwrite - for single bucket case this is fine,
            # for multi-bucket we'd want weighted averaging but that's a future enhancement)
            aggregated_metrics.update(bucket_metrics)

        # Aggregate per-sequence outputs back into per-request results
        all_results: dict[str, types.ForwardBackwardOutput | types.ErrorResponse] = {}

        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []

            for orig_idx in range(start_idx, end_idx):
                if orig_idx not in per_seq_outputs:
                    raise RuntimeError(
                        f"Missing output for sequence {orig_idx} in request {request_id}. "
                        f"This indicates a bug in bucket splitting/merging."
                    )
                loss_fn_outputs.append(per_seq_outputs[orig_idx])

            all_results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics=aggregated_metrics,
            )

        return all_results

    def process_optim_step(
        self,
        model_id: str,
        adapter_index: int,
        request_data: types.OptimStepInput,
    ) -> types.OptimStepOutput:
        """Process an optim_step request."""
        if self.accumulated_grads.count[0] == 0:
            log.warning("no accumulated gradients, skipping optimizer step", component="maxtext", model_id=model_id)
            return types.OptimStepOutput()

        optimizer = self.optimizers[model_id]
        hp = optimizer.opt_state.hyperparams
        hp["learning_rate"][...] = request_data.adam_params.learning_rate
        hp["b1"][...] = request_data.adam_params.beta1
        hp["b2"][...] = request_data.adam_params.beta2
        hp["eps"][...] = request_data.adam_params.eps

        mean_grads = self.accumulated_grads.get_mean()

        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
            self._optim_step(self.model, optimizer, mean_grads)

        self.accumulated_grads = self.accumulated_grads.reset()
        log.info("applied optimizer step", component="maxtext", model_id=model_id)

        return types.OptimStepOutput()

    def process_forward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward-only requests - not implemented for MaxText."""
        raise NotImplementedError("Forward-only pass not yet implemented for MaxText backend")

    def process_sample_batch(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process sample requests - not implemented for MaxText."""
        raise NotImplementedError("Sampling not yet implemented for MaxText backend")

    def save_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save training checkpoint as tar.gz to GCS.

        Serialization: flax.training.checkpoints (msgpack).
        Produces a file "checkpoint_0" inside a tar.gz at output_path.
        Contents: {lora_weights: jax arrays, optimizer_state: nnx.State leaves, lora_config: dict}.
        """
        with pack_and_upload(output_path) as temp_dir:
            checkpoint_data = self.extract_checkpoint_data(model_id, models)
            checkpoints.save_checkpoint(
                target=checkpoint_data,
                ckpt_dir=temp_dir,
                step=0,
                prefix="checkpoint_",
                overwrite=True,
            )
            log.info("saved training checkpoint", component="maxtext", model_id=model_id, output_path=str(output_path))

    def extract_checkpoint_data(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract LoRA weights + optimizer state as a plain dict for serialization.

        Returns a dict compatible with flax.training.checkpoints.save_checkpoint:
        - "lora_weights": copied jax arrays (tree structure mirrors self.lora_params)
        - "optimizer_state": copied nnx.State leaves (tree structure mirrors nnx.state(optimizer))
        - "lora_config": plain python dict from pydantic model_dump()

        Arrays are copied so the checkpoint is independent from live state (which gets zeroed on eviction).
        Also used as the `target` shape reference for restore_checkpoint on the load path.
        """
        # Copy arrays to avoid caching references that get zeroed
        lora_weights_copy = jax.tree.map(jnp.copy, self.lora_params)
        optimizer_state_copy = jax.tree.map(jnp.copy, nnx.state(self.optimizers[model_id]))
        return {
            "lora_weights": lora_weights_copy,
            "optimizer_state": optimizer_state_copy,
            "lora_config": models[model_id].lora_config.model_dump(),
        }

    def insert_checkpoint_data(
        self,
        model_id: str,
        checkpoint_data: dict,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Load checkpoint data into live model state.

        checkpoint_data leaves are numpy arrays (from msgpack deserialization via
        flax.training.checkpoints.restore_checkpoint) or jax arrays (from in-memory cache).
        Each leaf is resharded via jax.device_put to match the current optimizer/lora sharding,
        then written in-place via nnx.update.
        """
        optimizer = self.optimizers[model_id]

        # Reshard cached weights to match current model sharding
        def reshard_to_match(cached, current):
            """Reshard cached array to match current array's sharding."""
            sharding = current.sharding
            return jax.device_put(cached, sharding)

        resharded_lora = jax.tree.map(
            reshard_to_match, checkpoint_data["lora_weights"], self.lora_params
        )
        resharded_optim = jax.tree.map(
            reshard_to_match, checkpoint_data["optimizer_state"], nnx.state(optimizer)
        )

        # Update model state
        nnx.update(self.lora_params, resharded_lora)
        nnx.update(nnx.state(optimizer), resharded_optim)
        # Sync model with updated lora_params
        self.model = nnx.merge(self.graphdef, self.lora_params, self.non_lora_params)
        log.info("restored checkpoint data", component="maxtext", model_id=model_id)

    def save_sampler_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save sampler checkpoint in HuggingFace PEFT format as tar.gz."""
        with pack_and_upload(output_path) as temp_dir:
            convert_maxtext_lora_to_hf(
                lora_state=self.lora_params,
                output_path=temp_dir,
                base_model_name=self.config.base_model,
                lora_rank=self.maxtext_config.lora_rank,
                lora_alpha=self.maxtext_config.lora_alpha,
            )
        log.info("saved lora sampler checkpoint", component="maxtext", model_id=model_id, output_path=str(output_path))

    def extract_sampler_weights(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract sampler weights."""
        return {
            "lora_params": self.lora_params,
            "lora_rank": self.maxtext_config.lora_rank,
            "lora_alpha": self.maxtext_config.lora_alpha,
        }

    def insert_sampler_weights(
        self,
        model_id: str,
        checkpoint_id: str,
        weights_data: dict,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert sampler weights - not implemented for MaxText."""
        raise NotImplementedError("Loading sampler weights not yet implemented for MaxText backend")


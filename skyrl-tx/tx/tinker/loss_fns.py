"""Loss functions for training."""

import jax
import jax.numpy as jnp


def safe_loss_mask(loss_output: jax.Array, loss_mask: jax.Array) -> jax.Array:
    "Strongly mask the loss_output to 0.0 if the loss_mask is zero."
    return jnp.where(loss_mask != 0.0, loss_mask * loss_output, jnp.zeros_like(loss_output))


def cross_entropy_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    sampling_logprobs: jax.Array,
    advantages: jax.Array,
    clip_low: jax.Array,  # TODO: must fix this slop - unused param added for uniform signature across loss fns
    clip_high: jax.Array,  # TODO: must fix this slop - unused param added for uniform signature across loss fns
) -> jax.Array:
    "Standard cross-entropy loss (i.e., negative log-likelihood)."
    return -safe_loss_mask(target_logprobs, loss_mask)


def importance_sampling_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    sampling_logprobs: jax.Array,
    advantages: jax.Array,
    clip_low: jax.Array,  # TODO: must fix this slop - unused param added for uniform signature across loss fns
    clip_high: jax.Array,  # TODO: must fix this slop - unused param added for uniform signature across loss fns
) -> jax.Array:
    "Importance sampling loss with target_logprobs from learner policy and sampling_logprobs from sampling policy."
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    return -safe_loss_mask(prob_ratio * advantages, loss_mask)


def ppo_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    sampling_logprobs: jax.Array,
    advantages: jax.Array,
    clip_low: jax.Array,
    clip_high: jax.Array,
) -> jax.Array:
    "PPO style clipped version of the importance sampling loss."
    prob_ratio = jnp.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = jnp.clip(prob_ratio, clip_low, clip_high)
    unclipped = prob_ratio * advantages
    clipped = clipped_ratio * advantages
    return -safe_loss_mask(jnp.minimum(unclipped, clipped), loss_mask)


def mis_importance_sampling_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    sampling_logprobs: jax.Array,
    advantages: jax.Array,
    clip_low: jax.Array,
    clip_high: jax.Array,
) -> jax.Array:
    """Masked Importance Sampling (MIS) loss - masks entire sequence if ratio exceeds threshold.

    Unlike PPO which clips per-token ratios, MIS computes a sequence-level importance ratio
    and completely zeros the loss for sequences where the ratio exceeds bounds.
    This prevents biased gradients from highly divergent sequences.

    Args:
        clip_low: Lower bound for sequence ratio (e.g., 0.125). Sequences with ratio < clip_low are masked.
        clip_high: Upper bound for sequence ratio (e.g., 3.0). Sequences with ratio > clip_high are masked.

    Reference: "Defeating the Training-Inference Mismatch via FP16" (arxiv 2510.26788)
    """
    # Per-token log ratio
    log_ratio = target_logprobs - sampling_logprobs

    # Compute sequence-level log ratio (sum of masked log ratios)
    # Clamp individual log ratios for numerical stability before summing
    safe_log_ratio = jnp.clip(log_ratio, -20.0, 20.0)
    seq_log_ratio = (safe_log_ratio * loss_mask).sum()

    # Exponentiate to get sequence importance ratio
    seq_ratio = jnp.exp(seq_log_ratio)

    # MIS mask: keep sequence only if clip_low <= ratio <= clip_high
    # Use stop_gradient to prevent gradient flow through the mask
    in_bounds = (seq_ratio >= clip_low) & (seq_ratio <= clip_high)
    mis_mask = jax.lax.stop_gradient(in_bounds.astype(jnp.float32))

    # Per-token importance ratio for the loss
    prob_ratio = jnp.exp(log_ratio)

    # Apply MIS mask to entire sequence
    # If mis_mask is 0, all per-token losses become 0
    return -safe_loss_mask(mis_mask * prob_ratio * advantages, loss_mask)


def token_mis_importance_sampling_loss(
    target_logprobs: jax.Array,
    loss_mask: jax.Array,
    sampling_logprobs: jax.Array,
    advantages: jax.Array,
    clip_low: jax.Array,
    clip_high: jax.Array,
) -> jax.Array:
    """Token-level Masked Importance Sampling (MIS) loss.

    Unlike seq_mis which computes a sequence-level ratio and masks entire sequences,
    token_mis computes per-token ratios and masks only individual tokens that exceed bounds.
    This provides finer-grained control while still zeroing (not clipping) divergent tokens.

    Args:
        clip_low: Lower bound for per-token ratio. Tokens with ratio < clip_low are masked.
        clip_high: Upper bound for per-token ratio. Tokens with ratio > clip_high are masked.
    """
    # Per-token log ratio
    log_ratio = target_logprobs - sampling_logprobs

    # Per-token importance ratio
    prob_ratio = jnp.exp(log_ratio)

    # Token-level MIS mask: keep token only if clip_low <= ratio <= clip_high
    # Use stop_gradient to prevent gradient flow through the mask
    in_bounds = (prob_ratio >= clip_low) & (prob_ratio <= clip_high)
    token_mis_mask = jax.lax.stop_gradient(in_bounds.astype(jnp.float32))

    # Apply token-level mask - only divergent tokens are zeroed
    return -safe_loss_mask(token_mis_mask * prob_ratio * advantages, loss_mask)


# Map from string names to loss functions
# The ordering of this map determines the indices used in jax.lax.switch
LOSS_FUNCTION_MAP = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
    "seq_mis": mis_importance_sampling_loss,
    "token_mis": token_mis_importance_sampling_loss,
}

# Map from loss function name to index (for jax.lax.switch)
LOSS_TYPES = {name: idx for idx, name in enumerate(LOSS_FUNCTION_MAP.keys())}

# List of loss functions in order (for jax.lax.switch)
LOSS_FUNCTIONS = list(LOSS_FUNCTION_MAP.values())

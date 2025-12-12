# Segment-at-a-Time Weight Loading Design

## Problem
- MoE weights: 6GB total for `[6,8,128,2048,192]`
- Goal: Only 1GB (one segment `[8,...]`) in HBM at a time

## Approach: Sequential JIT per Segment with Custom VJP

### Key Insight
- **Hidden states**: Must flow through all segments (sequential dependency)
- **Weights**: Each segment's weights are independent

### Forward Pass (Python loop)
```python
def segmented_forward(initial_carry, all_segment_layers):
    """
    all_segment_layers: list of 6 segment layer states, each [8, ...]
    """
    carry = initial_carry
    boundary_carries = [carry]  # Save for backward

    for seg_idx in range(num_segments):
        # Only this segment's weights are in the JIT
        segment_layers = all_segment_layers[seg_idx]

        @jax.jit
        @jax.checkpoint  # Recompute during backward
        def process_segment(c, layers):
            return nnx.scan(scan_fn, length=8)(c, layers)

        carry, _ = process_segment(carry, segment_layers)
        boundary_carries.append(carry)

    return carry, boundary_carries
```

### Backward Pass (reverse Python loop)
```python
def segmented_backward(grad_output, boundary_carries, all_segment_layers):
    """
    Compute gradients flowing through segments in reverse.
    """
    grad_carry = grad_output
    all_grad_weights = []

    for seg_idx in reversed(range(num_segments)):
        segment_layers = all_segment_layers[seg_idx]  # Load [8, ...]
        input_carry = boundary_carries[seg_idx]

        # Compute gradients for this segment
        def segment_fn(c, layers):
            return process_segment(c, layers)[0]

        # Get gradients w.r.t. carry and weights
        grad_fn = jax.grad(segment_fn, argnums=(0, 1), has_aux=False)
        grad_carry, grad_weights = grad_fn(input_carry, segment_layers)

        all_grad_weights.insert(0, grad_weights)
        # grad_carry flows to previous segment

    return all_grad_weights, grad_carry
```

## Implementation Options

### Option 1: Manual Loop (Simplest)
- Pro: Clear control over memory
- Con: Breaks `nnx.value_and_grad` pattern, need custom training loop

### Option 2: Custom VJP on `segmented_forward`
```python
@jax.custom_vjp
def forward_all_segments(carry, layers):
    # Forward: process all segments
    ...

def forward_fwd(carry, layers):
    result, boundary_carries = forward_all_segments_impl(carry, layers)
    return result, (boundary_carries, layers)

def forward_bwd(res, g):
    boundary_carries, layers = res
    # Backward: load one segment at a time
    ...
```
- Pro: Works with standard JAX autodiff
- Con: More complex implementation

### Option 3: Use `jax.lax.scan` with `xs` as indices
```python
# Keep layers as [48, ...], pass indices to scan
def outer_body(carry, segment_idx):
    # Dynamic slice to get [8, ...] - XLA might optimize
    segment_layers = jax.lax.dynamic_slice(all_layers, [segment_idx * 8, ...], [8, ...])
    return process_segment(carry, segment_layers)

jax.lax.scan(outer_body, initial_carry, jnp.arange(6))
```
- Pro: Single JIT, might let XLA optimize
- Con: Research shows XLA doesn't optimize this well (100x gradient slowdown)

## Recommended Implementation

**Phase 1: Restructure scan to use segment indices**

Modify `qwen3.py` to:
1. Keep `self.layers` in `[48, ...]` format (not pre-reshaped)
2. Outer scan receives segment indices `[0,1,2,3,4,5]`
3. Inside checkpoint boundary, slice out `[8,...]` segment

**Phase 2: If Phase 1 doesn't help, implement Option 2 (custom_vjp)**

This gives explicit control over when segment weights are loaded during backward.

## Memory Comparison

| Approach | Forward HBM | Backward HBM | Complexity |
|----------|-------------|--------------|------------|
| Current (all segments) | 6GB | 6GB + remat | Low |
| Segment indices + slice | 6GB (XLA keeps all) | 6GB | Low |
| Custom VJP streaming | 1GB | 1GB | High |
| Manual loop | 1GB | 1GB | Medium |

## Key Files to Modify

1. `tx/models/qwen3.py`: Change scan to use segment indices
2. `flax/nnx/transforms/iteration.py`: Modify segment_length to use indices
3. `tx/tinker/engine.py`: Potentially restructure training loop for manual approach

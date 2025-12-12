# NNX remat_scan Implementation Notes

## Problem Summary

When using `nnx.scan` for transformer layer iteration with gradient computation, JAX's autodiff creates stacked intermediate tensors for the backward pass. For a 48-layer model, this results in significant memory allocation that causes OOM errors on TPUs.

## What We Implemented

### Approach: Add `segment_length` Parameter to `nnx.scan`

**File:** `flax/flax/nnx/transforms/iteration.py`

We added a `segment_length` parameter that enables nested scan with checkpointing:

```python
@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry, segment_length=8)
def apply_layers(carry, layer):
    return layer(carry)
```

### Implementation Details

1. **Parameter added to function signature** (lines 1126-1172):
   - Added `segment_length: int | None = None` to overloads and main function
   - Added documentation explaining memory/compute tradeoff

2. **Segmented scan logic** (lines 1355-1420):
   ```python
   if segment_length is not None and actual_length is not None:
       num_segments = actual_length // segment_length

       # Reshape inputs: [length, ...] -> [num_segments, segment_length, ...]
       def reshape_for_segments(x):
           if hasattr(x, 'shape') and hasattr(x, 'reshape') and hasattr(x, 'ndim'):
               if x.ndim > 0:
                   return x.reshape((num_segments, segment_length) + x.shape[1:])
           return x

       scan_in_segmented = jax.tree.map(reshape_for_segments, scan_in)

       # Nested scan structure
       def outer_scan_fn(carry, segment_scan_in):
           @jax.checkpoint(policy=jax.checkpoint_policies.nothing_saveable)
           def process_segment(c, seg_in):
               c_out, seg_out = jax.lax.scan(scan_fn, c, seg_in, length=segment_length, ...)
               return c_out, seg_out

           carry_out, segment_out = process_segment(carry, segment_scan_in)
           return carry_out, segment_out

       # Outer scan over segments
       carry_out, scan_out_segmented = jax.lax.scan(
           outer_scan_fn, carry, scan_in_segmented, length=num_segments, ...
       )

       # Reshape outputs back: [num_segments, segment_length, ...] -> [length, ...]
       scan_out = jax.tree.map(reshape_from_segments, scan_out_segmented)
   ```

## Results So Far

### The segmentation IS working:
- Memory profile shows shapes like `bf16[6,8,128,2048,192]` (6 segments × 8 layers)
- The nested `checkpoint/while/body/closed_call/checkpoint/while` structure appears in traces

### BUT memory is NOT reduced:
- XLA creates `.remat_compressed` copies of parameters
- Total memory usage increased rather than decreased
- Example from memory profile:
  ```
  1. Size: 6.00G - bf16[6,8,128,2048,192] - copy.331 = copy(param.39)
  2. Size: 6.00G - bf16[6,8,128,2048,192] - copy.330 = copy(param.38)
  3. Size: 4.50G - bf16[6,8,128,2048,192] - copy.331.remat_compressed
  4. Size: 4.50G - bf16[6,8,128,2048,192] - copy.330.remat_compressed
  ```

## What We Learned About Linen's `remat_scan`

### Linen's Implementation (flax/core/lift.py:1695-1769)

```python
def remat_scan(body_fn, lengths, policy=None, variable_broadcast=False,
               variable_carry=False, variable_axes={True: 0}, split_rngs={True: True}):
    scan_fn = functools.partial(scan, variable_broadcast=variable_broadcast, ...)

    if len(lengths) == 1:
        def wrapper(scope, carry):
            return body_fn(scope, carry), ()
        fn = lambda scope, c: scan_fn(wrapper, length=lengths[0])(scope, c)[0]
    else:
        @functools.partial(remat, policy=policy, prevent_cse=False)
        def inner_loop(scope, carry):
            carry = remat_scan(body_fn, lengths[1:], policy, ...)(scope, carry)
            return carry, ()

        fn = lambda scope, c: scan_fn(inner_loop, length=lengths[0])(scope, c)[0]
    return fn
```

### Key Differences from Our Approach

| Aspect | Linen's `remat_scan` | Our NNX Implementation |
|--------|---------------------|----------------------|
| **Parameters** | BROADCAST via `variable_broadcast` - same params accessible to all iterations | SCANNED - params are stacked `[num_layers, ...]` and sliced per iteration |
| **Param location** | "Outside" the scan, accessed via scope | "Inside" the scan as `scan_in` |
| **Remat scope** | Applied to `inner_loop` which contains recursive `remat_scan` | Applied to `process_segment` which contains inner scan |
| **Nesting** | Recursive - can have arbitrary depth via `lengths=(10, 10, 10)` | Two-level - outer segments, inner within-segment |
| **CSE** | `prevent_cse=False` explicitly set | Not specified |

### The Fundamental Problem

**Linen's approach:** Parameters are broadcast, meaning they exist ONCE and each iteration accesses them via scope. The remat only needs to save/restore the hidden state (carry), not the parameters.

**Our NNX approach:** Parameters are part of `scan_in`, meaning:
1. They get reshaped to `[num_segments, segment_length, ...]`
2. Each segment gets its slice of parameters
3. When `jax.checkpoint` wraps `process_segment`, the parameters (`seg_in`) are inputs
4. XLA sees parameters used in both forward and backward (for remat recomputation)
5. XLA creates `.remat_compressed` copies of parameters

### Why Our Approach Causes Parameter Duplication

1. In `outer_scan_fn`, we pass `segment_scan_in` (containing parameters) to `process_segment`
2. `process_segment` is checkpointed with `nothing_saveable` policy
3. During backward pass, JAX needs to recompute `process_segment`
4. To recompute, JAX needs `seg_in` (the parameters for that segment)
5. XLA creates compressed copies to have parameters available for recomputation

The `nothing_saveable` policy doesn't help because:
- It controls what RESIDUALS are saved (intermediate values)
- Parameters are INPUTS, not residuals
- XLA still needs parameters available for the recomputation phase

## What Needs to Change

To properly implement `remat_scan` for NNX, we likely need to:

1. **Restructure parameter handling:**
   - Keep parameters "outside" the checkpoint boundary
   - Pass only the hidden state (carry) through the checkpointed region
   - Have parameters accessed via some mechanism that doesn't require saving them

2. **Match Linen's broadcast pattern:**
   - Linen's `variable_broadcast` keeps parameters constant across iterations
   - In NNX, we'd need a way to broadcast stacked params without including them in checkpoint

3. **Possibly use closure capture:**
   - Define the scan body as a closure that captures parameters
   - Only pass the carry through the checkpoint
   - But this may have issues with JAX tracing...

## References

- Flax Linen `remat_scan`: `flax/core/lift.py:1695-1769`
- NNX scan implementation: `flax/nnx/transforms/iteration.py:1158-1420`
- JAX checkpoint: https://jax.readthedocs.io/en/latest/jax.checkpoint.html
- Original research doc: `SkyRL/skyrl-tx/docs/scan_remat_research.md`

## Next Steps

1. Study how Linen's `variable_broadcast` works in detail
2. Understand how to achieve similar "parameter broadcast" semantics in NNX
3. Possibly restructure to separate parameter access from checkpoint region
4. Consider whether NNX's stacked-params pattern is fundamentally incompatible with remat_scan

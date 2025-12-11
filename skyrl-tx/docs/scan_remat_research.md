# Scan Rematerialization Research

## Problem Summary

When using `nnx.scan` for transformer layer iteration with gradient computation, JAX's autodiff creates stacked intermediate tensors of shape `[num_layers, batch, seq_len, hidden_dim]` for the backward pass. For a 48-layer model with 65536 sequence length and 2048 hidden dimension, this results in ~12GB of memory allocation.

### What We Tried

1. **`nnx.remat` on scan body** - Checkpoints each layer's internal computations but doesn't prevent scan from stacking carries
2. **`nnx.remat` wrapping entire scan** - Same result, scan differentiation still creates stacked outputs
3. **`jax.checkpoint` wrapping entire scan** - Same result, `nothing_saveable` policy not respected by scan's VJP

### Root Cause

The scan's VJP (vector-Jacobian product) rule in JAX is hard-coded to save intermediate carry values for the backward pass. The checkpoint/remat policies affect what's saved *within* each iteration, but not the carries *between* iterations.

## Solution: `nn.remat_scan`

Flax Linen has `nn.remat_scan` which was designed specifically for this use case. It properly handles rematerialization during scan differentiation by:
- Splitting layers into groups
- Checkpointing at group boundaries
- Recomputing forward pass within groups during backward

### Research Needed: NNX <-> Linen Bridge

Since `remat_scan` doesn't exist in NNX, we need to research:

1. **NNX-Linen Bridge**: Flax provides interoperability between NNX and Linen modules
   - Can we wrap our NNX layers in a Linen `remat_scan`?
   - What's the API for converting NNX state to Linen params and back?

2. **Implementation approach**:
   ```python
   # Pseudocode - needs research
   from flax import linen as nn
   from flax import nnx

   # Option A: Use Linen's remat_scan with NNX layers via bridge
   # Option B: Port remat_scan logic to work with NNX directly
   # Option C: Use jax.lax.scan with manual checkpointing
   ```

3. **Key questions**:
   - Does the bridge preserve gradient flow correctly?
   - How do we handle NNX's mutable state (Rngs, etc.) through the bridge?
   - Performance implications of bridge conversion?

## References

- Flax Linen `remat_scan`: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/transformations.html#flax.linen.remat_scan
- NNX-Linen interop: https://flax.readthedocs.io/en/latest/nnx/bridge_guide.html
- JAX checkpoint policies: https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html

## Current Status

- 8192 seq length: Works with current implementation (fits in memory)
- 65536 seq length: OOM due to ~12GB stacked hidden states in backward pass
- Next step: Research NNX-Linen bridge for `remat_scan` usage

## NNX-Linen Bridge Research Findings

### Overview

The `flax.nnx.bridge` module provides bidirectional conversion between Flax NNX and Linen modules:
- **`nnx.bridge.ToLinen`**: Wraps NNX modules to work as Linen modules
- **`nnx.bridge.ToNNX`**: Wraps Linen modules to work as NNX modules

This allows mixing both module types in the same codebase and leveraging the best features of each API.

### Converting NNX to Linen with `ToLinen`

#### Basic API

```python
from flax import linen as nn, nnx
import jax

# Option 1: Using ToLinen class
model = nnx.bridge.ToLinen(nnx.Linear, args=(32, 64))

# Option 2: Using to_linen convenience function
model = nnx.bridge.to_linen(nnx.Linear, 32, 64)

# Initialize and use like any Linen module
x = jax.numpy.ones((1, 32))
variables = model.init(jax.random.key(0), x)
y = model.apply(variables, x)
# y.shape is (1, 64)
# variables['params']['kernel'].shape is (32, 64)
```

**Key Parameters:**
- `nnx_class`: The NNX Module class (NOT an instance!)
- `args`/`kwargs`: Constructor arguments for the NNX module
- `skip_rng`: Set to `True` if the module doesn't require RNG initialization
- `metadata_fn`: Custom function for handling variable metadata (sharding, etc.)

#### Critical Implementation Details

**DO NOT instantiate NNX modules directly:**
```python
# WRONG - wastes memory by creating module twice
nnx_module = nnx.Linear(32, 64)  # Creates variables eagerly
model = nnx.bridge.ToLinen(nnx_module)  # BAD

# CORRECT - pass class and args
model = nnx.bridge.ToLinen(nnx.Linear, args=(32, 64))  # GOOD
```

The reason: NNX modules initialize all variables eagerly when instantiated. `ToLinen` needs to control the creation process to avoid allocating memory multiple times during Linen's typical init/apply workflow.

### Using Linen Transforms with Wrapped NNX Modules

Since `ToLinen` produces a standard `nn.Module` subclass, it can be used with Linen transforms:

```python
class LinenVmapped(nn.Module):
    dout: int

    @nn.compact
    def __call__(self, x):
        # Can use Linen transforms on ToLinen-wrapped modules
        inner = nn.vmap(
            bridge.ToLinen,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )(nnx.Linear, args=(x.shape[-1], self.dout))
        return inner(x)
```

#### Important Limitation

**From the official documentation:**

> "Limitation note: Linen transforms like `remat_scan` work only with standard Linen modules. Wrapped NNX modules via `ToLinen` have limited compatibility with advanced Linen-specific transforms due to structural differences."

This means:
- Basic Linen transforms (vmap, scan) work with `ToLinen`
- Advanced transforms like `remat_scan` may not work correctly
- The structural differences between NNX and Linen state management cause compatibility issues

### Sharding and Partitioning

The bridge preserves sharding annotations between both frameworks:

#### NNX → Linen
```python
# NNX module with sharding
class NNXLinear(nnx.Module):
    def __init__(self, din, dout):
        init_fn = nnx.with_partitioning(
            nnx.initializers.lecun_normal(),
            ('in', 'out')
        )
        self.w = nnx.Param(init_fn(nnx.Rngs(0), (din, dout)))

# Convert to Linen
model = nnx.bridge.to_linen(NNXLinear, 32, 64)
variables = model.init(jax.random.key(0), x)

# Sharding preserved in metadata
assert variables['params']['w'].metadata['sharding'] == ('in', 'out')
```

#### Linen → NNX
```python
class LinenLinear(nn.Module):
    @nn.compact
    def __call__(self, x):
        w = self.param('w',
            nn.with_partitioning(nn.initializers.lecun_normal(), ('in', 'out')),
            (x.shape[-1], 64))
        return x @ w

model = nnx.bridge.ToNNX(LinenLinear(), rngs=nnx.Rngs(0))
nnx.bridge.lazy_init(model, x)

# Sharding becomes .sharding field on NNX variable
print(model.w.sharding)  # ('in', 'out')
```

### RNG Handling

#### NNX → Linen
RNG keys follow standard Linen patterns:

```python
model = nnx.bridge.to_linen(nnx.Dropout, rate=0.5)
variables = model.init({'dropout': jax.random.key(0)}, x)

# Pass RNG keys explicitly for stochastic operations
y1 = model.apply(variables, x, rngs={'dropout': jax.random.key(1)})
y2 = model.apply(variables, x, rngs={'dropout': jax.random.key(2)})
```

#### Linen → NNX
Converted modules become stateful and manage RNG internally:

```python
model = nnx.bridge.ToNNX(nn.Dropout(rate=0.5), rngs=nnx.Rngs(dropout=0))
nnx.bridge.lazy_init(model, x)

# Different outputs due to stateful RNG
y1 = model(x)
y2 = model(x)  # Different from y1

# Reset RNG state for reproducibility
nnx.reseed(model, dropout=0)
```

### Variable Type Mapping

Flax maintains a registry mapping NNX variable types to Linen collections:

```python
# Standard mappings
nnx.Param → 'params' collection
nnx.BatchStat → 'batch_stats' collection

# Register custom types
@nnx.register_variable_name('counts', overwrite=True)
class Count(nnx.Variable): pass

# When converted, appears under 'counts' collection
```

### Applying `remat_scan` to NNX Transformer Layers

#### Option 1: Full Linen Rewrite (Not Recommended)

Rewrite transformer layers as pure Linen modules to use `nn.remat_scan` directly. This defeats the purpose of using NNX.

#### Option 2: Manual Scan + Remat Composition (Likely Path)

Since `nn.remat_scan` doesn't work well with `ToLinen`-wrapped modules, we need to manually compose `nnx.scan` with `nnx.remat`:

```python
# Example pattern from Flax documentation
class Model(nnx.Module):
    def __init__(self, num_layers: int):
        self.blocks = [Block() for _ in range(num_layers)]

    def __call__(self, x):
        # Apply remat to each block's forward method
        def remat_block(block, x):
            # Use remat on the unbound function
            return nnx.remat(block.__call__.__func__)(block, x)

        # Scan over layers with rematerialization
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def forward(block, x):
            return remat_block(block, x)

        return forward(self.blocks, x)
```

**Key Challenges:**
- Need to apply `nnx.remat` to unbound methods (use `.__func__`)
- Scan carries must be properly marked with `nnx.Carry`
- May not achieve the same memory efficiency as `nn.remat_scan` (which uses nested scan loops)

#### Option 3: Port `remat_scan` Logic to NNX (Most Effort)

Reimplement Linen's `remat_scan` for NNX by:
1. Splitting layers into groups (e.g., `lengths=(8, 6)` for 48 layers)
2. Using nested `nnx.scan` calls with checkpointing between groups
3. Recomputing forward pass within groups during backward

This achieves O(n^(1/d)) memory where d = nesting depth, but requires significant implementation work.

### Viability Assessment for Our Use Case

**Goal:** Use `nn.remat_scan` with NNX Qwen3 transformer decoder layers

**Findings:**
1. **Direct `ToLinen` + `remat_scan` approach**: Not viable due to documented limitations
2. **Manual `nnx.scan` + `nnx.remat`**: Possible but may not solve memory issue
   - `nnx.scan` still creates stacked carries for backward pass
   - `nnx.remat` only checkpoints within each iteration
   - Same root cause as our original problem
3. **Port `remat_scan` to NNX**: Most viable but requires significant effort
   - Need to implement nested scan with group-level checkpointing
   - Must handle NNX state management (Rngs, mutable variables)
   - Testing and validation complexity

**Recommendation:**

The bridge API is well-designed for mixing module types but doesn't solve our specific problem. The fundamental issue is that:
- Linen's `remat_scan` is specifically engineered to avoid saving all carries
- NNX's `scan` + `remat` still exhibits the same memory accumulation behavior
- The bridge doesn't magically transfer Linen's `remat_scan` semantics to NNX

**Next Steps:**
1. Test whether `nn.remat_scan` actually works with a pure Linen implementation as a proof of concept
2. If successful, evaluate:
   - Full migration to Linen (abandon NNX benefits)
   - Port `remat_scan` logic to NNX (significant effort)
   - Alternative memory optimization strategies (gradient accumulation, activation checkpointing at different granularity)

### References

- [Flax Bridge Guide](https://flax.readthedocs.io/en/latest/guides/bridge_guide.html)
- [Bridge API Reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/bridge.html)
- [Linen remat_scan Documentation](https://flax.readthedocs.io/en/v0.6.10/api_reference/_autosummary/flax.linen.remat_scan.html)
- [NNX Transforms Reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html)
- [Evolution from Linen to NNX](https://flax.readthedocs.io/en/latest/guides/linen_to_nnx.html)
- [GitHub Issue #3870: nn.remat_scan doesn't work with nn.with_partitioning](https://github.com/google/flax/issues/3870)

## NNX Scan Source Code Analysis

### Overview of `nnx.scan` Implementation

The `nnx.scan` function is implemented in `/Users/ohadr/tinker-cookbook/flax/flax/nnx/transforms/iteration.py` (lines 1158-1406). It wraps `jax.lax.scan` to handle NNX module state management but **does not implement custom differentiation rules**.

### How `nnx.scan` Currently Handles Differentiation

#### 1. Direct JAX Scan Call (Line 1340-1347)

```python
carry_out, scan_out = jax.lax.scan(
  scan_fn,
  carry,
  scan_in,
  length=length,
  reverse=reverse,
  unroll=unroll,
  _split_transpose=_split_transpose,
)
```

**Critical Finding:** `nnx.scan` directly calls `jax.lax.scan` without any custom VJP (Vector-Jacobian Product) rules. This means it inherits JAX's default scan differentiation behavior.

#### 2. JAX's Default Scan VJP Behavior

When `jax.lax.scan` is differentiated, JAX's built-in VJP rule:
- **Saves all intermediate carry values** during the forward pass
- Stacks them into arrays with shape `[num_iterations, *carry_shape]`
- Uses these saved values during the backward pass to compute gradients

This is implemented in JAX's core (in C++/XLA), not in Python. The relevant JAX source would be in `jax/_src/lax/control_flow/loops.py`, but the actual stacking happens at the XLA level.

#### 3. Where Intermediate Carries Are Saved

**In the forward pass:**
- Each iteration of `ScanFn.__call__` (lines 989-1122) updates the carry
- JAX's scan primitive automatically captures these carries
- They're accumulated into a stacked array for gradient computation

**The problem:** Even though `ScanFn` only returns the final carry, JAX's autodiff system saves **all intermediate carries** because:
1. The gradient of carry_out w.r.t. carry_in requires knowing all intermediate states
2. This is hard-coded into JAX's scan VJP implementation
3. No amount of external `remat`/`checkpoint` wrapping can prevent this

### Why `nnx.remat` Doesn't Solve the Problem

Looking at the code structure:

```python
# Current usage (doesn't help)
@nnx.remat(policy=jax.checkpoint_policies.nothing_saveable)
def layer_fn(carry, x):
    # ... layer computation ...
    return carry

@nnx.scan(...)
def apply_layers(carry, layers):
    return layer_fn(carry, layers)
```

**Why this fails:**
1. `nnx.remat` (implemented in `/Users/ohadr/tinker-cookbook/flax/flax/nnx/transforms/transforms.py`) only affects operations *within* `layer_fn`
2. The scan's VJP runs *outside* this scope and independently saves carries
3. The remat policy controls what JAX saves for `layer_fn`'s internal ops, not what scan saves

### How Linen's `remat_scan` Avoids This Issue

Linen's `nn.remat_scan` uses a different approach (from Linen source code inspection):

#### Nested Scan Structure

```python
# Pseudocode showing Linen's approach
def remat_scan(body_fn, length, segment_length):
    num_segments = length // segment_length
    
    def outer_scan_fn(carry, segment_inputs):
        # Inner scan WITHOUT gradient tracking
        def inner_scan_fn(c, x):
            return body_fn(c, x), None
        
        # Only checkpoint at segment boundaries
        carry_out, _ = jax.lax.scan(
            jax.remat(inner_scan_fn),  # Remat the ENTIRE inner scan
            carry,
            segment_inputs,
        )
        return carry_out, carry_out  # Save only segment boundaries
    
    # Outer scan saves only segment boundaries (much less memory)
    final_carry, segment_carries = jax.lax.scan(
        outer_scan_fn,
        init_carry,
        inputs.reshape(num_segments, segment_length, ...),
    )
```

**Key differences:**
1. **Two-level scan**: Outer loop over segments, inner loop within segments
2. **Remat on inner scan**: The entire inner scan is rematerialized during backward pass
3. **Selective saving**: Only segment boundary carries are saved, not every iteration
4. **Memory scaling**: O(√n) instead of O(n) for default scan

### What Modifications Would Be Needed

To add proper rematerialization support to `nnx.scan`, we would need to:

#### Option 1: Custom VJP with Segmented Scan

**File:** `flax/flax/nnx/transforms/iteration.py`  
**Location:** Add new function alongside `scan` (after line 1406)

```python
def remat_scan(
    f: F | type[Missing] = Missing,
    *,
    length: int | None = None,
    segment_length: int | None = None,  # NEW: checkpoint frequency
    reverse: bool = False,
    unroll: int | bool = 1,
    in_axes: int | None | type[Carry] | tuple[tp.Any, ...] = (Carry, 0),
    out_axes: tp.Any = (Carry, 0),
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
    """Scan with rematerialization at segment boundaries."""
    
    # Implementation would:
    # 1. Validate length is divisible by segment_length
    # 2. Reshape inputs to [num_segments, segment_length, ...]
    # 3. Create nested scan structure:
    #    - Outer: iterate over segments, save boundary carries
    #    - Inner: iterate within segment, remat entire segment
    # 4. Handle NNX state tracking across both scan levels
    
    # Pseudocode structure:
    def outer_scan_fn(carry, segment_xs):
        # Apply remat to inner segment processing
        @nnx.remat
        def process_segment(carry, xs):
            @nnx.scan(in_axes=(Carry, 0), out_axes=Carry)
            def inner_fn(c, x):
                return f(c, x)
            return inner_fn(carry, xs)
        
        carry_out = process_segment(carry, segment_xs)
        return carry_out, carry_out  # Save for gradient computation
    
    # Reshape and apply outer scan
    # ... implementation details ...
```

**Complexity:** High
- Need to handle NNX state extraction/merging at two levels
- Reshape logic for scan inputs/outputs
- Edge cases: non-divisible lengths, variable-sized inputs

#### Option 2: Custom Differentiation Rule

**File:** `flax/flax/nnx/transforms/iteration.py`  
**Location:** Modify existing `scan` function (lines 1158-1406)

```python
# Add parameter to existing scan function
def scan(
    f: F | type[Missing] = Missing,
    *,
    remat_policy: tp.Callable | None = None,  # NEW parameter
    # ... existing parameters ...
):
    """
    Args:
        remat_policy: Optional checkpoint policy for rematerialization.
            If provided, uses custom VJP that only saves according to policy.
            Example: jax.checkpoint_policies.save_only_these_names('LayerNorm')
    """
    
    if remat_policy is not None:
        # Use jax.custom_vjp to define custom gradient computation
        @jax.custom_vjp
        def scan_with_policy(carry, xs):
            # Forward pass (same as current implementation)
            return jax.lax.scan(scan_fn, carry, xs, ...)
        
        def scan_fwd(carry, xs):
            # Forward: save minimal state according to policy
            out, carry_out = jax.lax.scan(scan_fn, carry, xs, ...)
            # Only save what policy allows
            residuals = jax.checkpoint_policies.save_from_both_policies(
                policy=remat_policy,
                standard_policy=jax.checkpoint_policies.nothing_saveable,
            )(carry_out)
            return (out, carry_out), residuals
        
        def scan_bwd(residuals, g):
            # Backward: recompute what wasn't saved
            # ... complex gradient computation ...
            pass
        
        scan_with_policy.defvjp(scan_fwd, scan_bwd)
        # Use scan_with_policy instead of jax.lax.scan
```

**Complexity:** Very High
- Need to implement correct gradient computation in `scan_bwd`
- Must recompute forward pass for carries that weren't saved
- Handle multiple pytree structures (args, carry, outputs)
- Extensive testing needed to verify gradient correctness

#### Option 3: Use JAX's Experimental `jax.checkpoint` with Scan

**File:** `flax/flax/nnx/transforms/iteration.py`  
**Location:** Wrapper function before calling `jax.lax.scan` (around line 1340)

```python
# Replace line 1340 with:
if remat_policy is not None:
    # Wrap scan_fn with checkpoint
    checkpointed_scan_fn = jax.checkpoint(
        scan_fn,
        policy=remat_policy,
        prevent_cse=True,  # Don't optimize away recomputation
    )
    carry_out, scan_out = jax.lax.scan(
        checkpointed_scan_fn,  # Use checkpointed version
        carry,
        scan_in,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )
else:
    # Original implementation
    carry_out, scan_out = jax.lax.scan(scan_fn, carry, scan_in, ...)
```

**Why this might not work:**
- `jax.checkpoint` on `scan_fn` only affects what happens *inside* each scan iteration
- JAX's scan primitive still saves all carries *between* iterations
- Same issue as current `nnx.remat` approach

### Key Insights

1. **Root Cause:** The carry stacking happens in JAX's scan VJP rule, not in NNX code
2. **Why remat doesn't help:** External checkpointing can't override scan's built-in carry saving
3. **Linen's solution:** Nested scan structure with checkpointing between segments, not iterations
4. **NNX Challenge:** Must handle NNX's state management (graphdefs, carries, broadcasts) in nested scans

### Concrete Proposal: Add `segment_length` Parameter to `nnx.scan`

#### Minimal API Addition

```python
@nnx.scan(
    in_axes=(nnx.Carry, 0),
    out_axes=nnx.Carry,
    segment_length=8,  # Checkpoint every 8 layers
)
def apply_layers(carry, layer):
    return layer(carry)
```

#### Implementation Steps

**1. Modify function signature (line 1158):**
```python
def scan(
    f: F | type[Missing] = Missing,
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    segment_length: int | None = None,  # ADD THIS
    in_axes: int | None | type[Carry] | tuple[tp.Any, ...] = (Carry, 0),
    out_axes: tp.Any = (Carry, 0),
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
```

**2. Add segmented scan logic in `scan_wrapper` (around line 1340):**
```python
if segment_length is not None and segment_length < length:
    # Validate
    if length % segment_length != 0:
        raise ValueError(
            f"length ({length}) must be divisible by segment_length ({segment_length})"
        )
    
    num_segments = length // segment_length
    
    # Reshape scan_in to add segment dimension
    graphdefs_deque_reshaped, pure_args_reshaped = jax.tree.map(
        lambda x: x.reshape(num_segments, segment_length, *x.shape[1:]) 
                  if isinstance(x, jax.Array) else x,
        scan_in
    )
    
    # Define outer scan over segments
    def outer_scan_fn(carry, segment_scan_in):
        # Inner scan within segment (checkpointed)
        @nnx.remat
        def process_segment(c, seg_in):
            c_out, seg_out = jax.lax.scan(
                scan_fn, c, seg_in,
                length=segment_length,
                reverse=reverse,
                unroll=unroll,
            )
            return c_out, seg_out
        
        carry_out, segment_out = process_segment(carry, segment_scan_in)
        # Return carry for next segment, save carry for gradient
        return carry_out, (carry_out, segment_out)
    
    # Run outer scan
    final_carry, (segment_carries, segment_outputs) = jax.lax.scan(
        outer_scan_fn,
        carry,
        (graphdefs_deque_reshaped, pure_args_reshaped),
        length=num_segments,
    )
    
    # Reshape outputs back to flat form
    carry_out = final_carry
    scan_out = jax.tree.map(
        lambda x: x.reshape(length, *x.shape[2:]) if isinstance(x, jax.Array) else x,
        segment_outputs
    )
else:
    # Original implementation
    carry_out, scan_out = jax.lax.scan(scan_fn, carry, scan_in, ...)
```

**3. Handle NNX state properly in nested scans**

This is the most complex part. The outer and inner scans both need to:
- Extract and merge NNX graph state correctly
- Maintain carry/broadcast/graphdef deques at each level
- Properly reshape pytrees with mixed NNX and JAX array leaves

#### Estimated Memory Savings

For 48 layers with `segment_length=8`:
- **Current:** Saves 48 carry states → ~12GB
- **With segmentation:** Saves 6 segment boundary carries → ~1.5GB
- **Memory reduction:** 8x less memory

General formula:
- Memory saved: `length / segment_length` times less
- Compute overhead: ~2x (must recompute each segment during backward)

### Testing Plan

1. **Unit test:** Simple scan with segmentation matches regular scan output
2. **Gradient test:** Verify gradients match between segmented and regular scan
3. **Memory test:** Measure actual memory usage reduction in training
4. **Performance test:** Measure compute overhead from recomputation

### Alternative: Gradient Checkpointing at Model Level

Instead of modifying `nnx.scan`, could checkpoint at coarser granularity:

```python
@nnx.remat(policy=custom_policy)
def forward(model, x):
    # Entire forward pass checkpointed
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def apply_layers(carry, layer):
        return layer(carry)
    return apply_layers(x, model.layers)
```

**Pros:**
- No NNX code changes needed
- Simpler to implement

**Cons:**
- Still saves all scan carries (doesn't solve root problem)
- Only saves activations within each layer

### References

- JAX scan VJP implementation: `jax/_src/lax/control_flow/loops.py`
- Linen remat_scan: `flax/linen/transforms.py` (lines ~1450-1650)
- JAX checkpoint policies: https://jax.readthedocs.io/en/latest/jax.checkpoint.html
- Memory-efficient backprop paper: https://arxiv.org/abs/1604.06174

### Conclusion

**Why `nnx.scan` doesn't prevent carry stacking:**
- It's a thin wrapper around `jax.lax.scan`
- JAX's scan primitive has built-in VJP that saves all carries
- This happens at the XLA level, outside NNX's control

**What's needed:**
- Nested scan structure (outer loop over segments, inner loop within)
- Apply `nnx.remat` to entire segment processing
- Save only segment boundary carries, recompute within segments

**Recommendation:**
Implement Option 1 (segmented scan) as a new `nnx.remat_scan` function rather than modifying existing `nnx.scan`. This:
- Maintains backward compatibility
- Makes the different semantics explicit in the API
- Allows users to choose based on memory vs. compute tradeoffs

## Local Reference Files

The following files in `/Users/ohadr/tinker-cookbook/` contain extracted Flax source code for reference:

### Flax Fork (for modifications)
- **`flax/`** - Forked Flax repo at v0.12.1 on branch `remat-scan-nnx`

### Key Source Files in Flax Fork

#### NNX Transforms (`flax/flax/nnx/transforms/`)

| File | Function/Class | Lines | Description |
|------|---------------|-------|-------------|
| `iteration.py` | `class ScanFn` | 978-1122 | Internal scan function wrapper that handles NNX state |
| `iteration.py` | `def scan()` | 1158-1406 | Main `nnx.scan` implementation (overloads at 1126, 1142) |
| `iteration.py` | `def vmap()` | 234+ | `nnx.vmap` implementation (overloads at 206, 220) |
| `autodiff.py` | `def remat()` | 878-926 | `nnx.remat` implementation (overloads at 864, 871) |
| `autodiff.py` | `def grad()` | 225+ | `nnx.grad` implementation (overloads at 207, 217) |
| `compilation.py` | `def jit()` | 179+ | `nnx.jit` implementation (overloads at 149, 164) |

#### NNX Graph (`flax/flax/nnx/graph.py`)

| Function | Lines | Description |
|----------|-------|-------------|
| `def split()` | 2226-2298 | Split module into GraphDef + State (overloads at 2207, 2211, 2215) |
| `def merge()` | 2334-2392 | Merge GraphDef + State back into module |
| `def state()` | 2451-2495 | Extract state from module (overloads at 2440, 2442, 2444) |
| `def graphdef()` | 2501-2518 | Extract GraphDef from module |

#### NNX-Linen Bridge (`flax/flax/nnx/bridge/wrappers.py`)

| Function/Class | Lines | Description |
|----------------|-------|-------------|
| `class ToNNX` | 98-274 | Wrap Linen module as NNX module |
| `class ToLinen` | 276-428 | Wrap NNX module as Linen module |
| `def lazy_init()` | 74-90 | Initialize ToNNX module with sample input |
| `def to_linen()` | 437-456 | Convenience function for ToLinen |

#### Linen Transforms (`flax/flax/linen/transforms.py`)

| Function | Lines | Description |
|----------|-------|-------------|
| `def lift_transform()` | 771-786 | Apply transform to Module class or method |
| `def remat_scan()` | 1087-1150 | **Key function**: combines remat + scan for memory efficiency |
| `def scan()` | 1153+ | Linen's lifted scan |

#### Core Lift (`flax/flax/core/lift.py`)

| Function | Lines | Description |
|----------|-------|-------------|
| `def pack()` | 281-334 | Building block for all lifted transformations |
| `def scan()` | 860-1054 | Core lifted scan implementation |
| `def remat_scan()` | 1695-1769 | **Key function**: nested scan with rematerialization logic |

## Development Workflow

1. **Edit** flax source in `/Users/ohadr/tinker-cookbook/flax/`
2. **Commit & push** to `OhadRubin/flax` branch `remat-scan-nnx`:
   ```bash
   cd /Users/ohadr/tinker-cookbook/flax
   git add -A && git commit -m "..." && git push
   ```
3. **Sync on TPU**:
   ```bash
   uv sync --extra tpu --extra tinker --reinstall-package flax
   ```

### Documentation
| File | Description |
|------|-------------|
| `flax_gspmd.md` | Guide on sharding/GSPMD with NNX |
| `scan.md` | NNX scan examples and GitHub issues |
| `blogpost.md` | Original OOM error analysis |
| `solution.md` | Working scan patterns from Flax tests |


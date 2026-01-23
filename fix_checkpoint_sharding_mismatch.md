
./resume.sh --no-server --run-dir /tmp/tinker-examples/rl-treebench_evalbox-30b-a3b-20260122-154327
TODO for CLAUDE:

## Table of Contents

1. [Context: Resume Flow That Led to This Error](#context-resume-flow-that-led-to-this-error)
2. [Root Cause Theory](#root-cause-theory)
3. [Key Insight](#key-insight)
4. [Proposed Fix Direction](#proposed-fix-direction)
5. [Client-Side Traceback](#client-side-traceback)
6. [Server-Side Logs (TPU)](#server-side-logs-tpu)
7. [Broken Resume: MaxText Backend Never Implemented Training Checkpoints](#broken-resume-maxtext-backend-never-implemented-training-checkpoints)
   - [Problem](#problem)
   - [The Real Root Cause: Save/Load Format Mismatch](#the-real-root-cause-saveload-format-mismatch)
   - [Two Save Paths (disambiguation)](#two-save-paths-disambiguation)
   - [Save Path (maxtext.py:654-669)](#save-path-maxtextpy654-669)
   - [Load Path (engine.py:525-536)](#load-path-enginepy525-536)
   - [Contrast with Native Backend (native.py:620-636)](#contrast-with-native-backend-nativepy620-636)
   - [Consequence](#consequence)
   - [So How Did The Error Occur?](#so-how-did-the-error-occur)
   - [How The Problem Was Masked](#how-the-problem-was-masked)
   - [Open Questions](#open-questions)
8. [Context: Resume Flow That Led to This Error (detailed)](#context-resume-flow-that-led-to-this-error-1)
9. [Relevant Code Locations](#relevant-code-locations)
10. [Secondary Issue: Scalar Sharding](#secondary-issue-scalar-sharding-if-optimizer-state-is-validly-loaded)
11. [Why Both Fixes Are Needed Together](#why-both-fixes-are-needed-together)
12. [Client-Side Traceback (detailed)](#client-side-traceback-1)
13. [Server-Side Logs (detailed)](#server-side-logs-tpu-1)
14. [Load Checkpoint Codepath](#load-checkpoint-codepath)
    - [Current Codepath (Broken)](#current-codepath-broken)
    - [Desired Codepath (Fixed)](#desired-codepath-fixed)
    - [Summary of Changes Required (Load Side)](#summary-of-changes-required-load-side)
15. [Save Checkpoint Codepath: Current vs Desired](#save-checkpoint-codepath-current-vs-desired)
    - [Current Codepath (BROKEN for resume)](#current-codepath-broken-for-resume)
    - [Desired Codepath (resume-compatible)](#desired-codepath-resume-compatible)
    - [Prerequisite: register_model mesh context](#prerequisite-register_model-mesh-context-maxtextpy287-293)
    - [Reference: Native Backend](#reference-native-backend-working-implementation-nativepy620-636)
    - [Files to Modify](#files-to-modify)
    - [Note on save_sampler_checkpoint](#note-on-save_sampler_checkpoint-maxtextpy722-737)
    - [Note for Load Checkpoint Agent](#note-for-load-checkpoint-agent)

<start>

## Context: Resume Flow That Led to This Error

We resumed a crashed training run by reusing the same `log_path`. The flow:

1. Client-side `checkpoints.jsonl` at `/tmp/tinker-examples/rl-treebench_evalbox-30b-a3b-20260122-154327/` had:
   ```json
   {"name": "000020", "batch": 20, "state_path": "tinker://model_9396ef01/weights/000020", "sampler_path": "tinker://model_9396ef01/000020"}
   ```

2. The training code (`tinker_cookbook/rl/train.py:1302-1316`) found this checkpoint and called:
   ```python
   training_client = await service_client.create_training_client_from_state_with_optimizer_async(
       resume_info["state_path"]  # "tinker://model_9396ef01/weights/000020"
   )
   ```

3. The SDK (`.venv/.../tinker/lib/public_interfaces/service_client.py:323-342`):
   - Called `POST /api/v1/weights_info` with the old tinker path → got base_model + lora_rank
   - Called `POST /api/v1/create_model` → created NEW model `model_2596ee3b`
   - Called `POST /api/v1/load_weights` → loaded checkpoint from GCS into the new model

4. On the server (`SkyRL/skyrl-tx/tx/tinker/engine.py:515-539`), `process_load_weights`:
   - Found `/mnt/gcs_bucket/lora-experiments-checkpoints/model_9396ef01/000020.tar.gz`
   - Called `restore_checkpoint` (orbax) to deserialize
   - Called `backend.insert_checkpoint_data(model_id, checkpoint, self.models)`

5. `insert_checkpoint_data` (`SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py:690-720`) reshards and loads:
   ```python
   def reshard_to_match(cached, current):
       sharding = current.sharding
       return jax.device_put(cached, sharding)

   resharded_optim = jax.tree.map(
       reshard_to_match, checkpoint_data["optimizer_state"], nnx.state(optimizer)
   )
   nnx.update(nnx.state(optimizer), resharded_optim)
   ```

6. `forward_backward` succeeded (line 40: "Batch [0:2] forward-backward time: 28.655 sec").
   Then `optim_step` failed with the device mismatch error.

## Root Cause Theory

The checkpoint's `extract_checkpoint_data` (`maxtext.py:671-686`) saves:
```python
optimizer_state_copy = jax.tree.map(jnp.copy, nnx.state(self.optimizers[model_id]))
```

When serialized to disk via orbax and restored, scalar values (like the optimizer step counter
`optimizer.states[0][0]`, an `int32[]`) lose their mesh sharding — they come back as plain
arrays on device 0 only.

The `reshard_to_match` function tries to fix this by reading `.sharding` from the freshly-created
optimizer's state and calling `jax.device_put`. But either:
- The fresh optimizer's scalar doesn't expose a proper multi-device sharding (scalars may default
  to single-device), so `reshard_to_match` is a no-op, OR
- `nnx.update` doesn't propagate the resharded value into the actual optimizer state that the
  JIT'd `optim_step` function captures

The error confirms this: `optimizer.states[0][0]` has `device ids [0]` but the jitted `optim_step`
was compiled expecting mesh `device ids [0, 2, 1, 3]` (the full 4-chip v5p-8 TPU mesh).

## Key Insight

This is a server-side bug in `SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py`. The `insert_checkpoint_data`
function doesn't properly handle scalar optimizer states (step counters) that need to be replicated
across the full mesh. The fix likely needs to ensure ALL restored optimizer leaves are placed on
the correct mesh sharding, including scalars that should be `NamedSharding(mesh, PartitionSpec())`
(fully replicated).

## Relevant Code Locations

- **Checkpoint save**: `SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py:671-686` (`extract_checkpoint_data`)
- **Checkpoint load + reshard**: `SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py:690-720` (`insert_checkpoint_data`)
- **optim_step dispatch**: `SkyRL/skyrl-tx/tx/tinker/engine.py:513` (`process_optim_step`)
- **optim_step JIT**: `SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py:633` — `self._optim_step(self.model, optimizer, mean_grads)`
- **_optim_step definition**: search for `_create_loss_and_grad_fn` in `maxtext.py`
- **Client resume logic**: `tinker_cookbook/rl/train.py:1302-1316`

## How the Optimizer Is Created (Fresh Model)

In `SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py:287-293`, `register_model`:
```python
def register_model(self, model_id: str, adapter_index: int, lora_config: types.LoraConfig) -> None:
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
    self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)
```

`nnx.Optimizer(self.model, tx, wrt=self.lora_filter)` initializes optimizer states (including
the `int32[]` step counter) derived from `self.model`, which is already sharded across the mesh.
So the fresh optimizer's step counter is replicated across all 4 TPU chips `[0, 2, 1, 3]`.

The JIT'd `optim_step` (line 257-264):
```python
def optim_step(model, optimizer, grads):
    optimizer.update(model, grads)

self._optim_step = nnx.jit(optim_step)
```

`nnx.jit` infers sharding from the arguments. On first call with a fresh optimizer, the step
counter is mesh-replicated, so the JIT compiles expecting mesh device ids `[0, 2, 1, 3]`.

After `insert_checkpoint_data` restores the optimizer state from disk, the step counter ends up
on device `[0]` only, causing the mismatch on the next `optim_step` call.

## Proposed Fix Direction

In `insert_checkpoint_data` (maxtext.py:690-720), after restoring the optimizer state, ensure
ALL scalar leaves are explicitly replicated across the mesh. Something like:

```python
from jax.sharding import NamedSharding, PartitionSpec as P

def reshard_to_match(cached, current):
    if hasattr(current, 'sharding'):
        return jax.device_put(cached, current.sharding)
    # Scalars that lost sharding: replicate across mesh
    return jax.device_put(cached, NamedSharding(self.mesh, P()))
```

Or alternatively, after `nnx.update`, re-shard the entire optimizer state to ensure consistency:
```python
nnx.update(nnx.state(optimizer), resharded_optim)
# Force all optimizer leaves onto the mesh
opt_state = nnx.state(optimizer)
resharded = jax.tree.map(
    lambda x: jax.device_put(x, NamedSharding(self.mesh, P())) if x.ndim == 0 else x,
    opt_state
)
nnx.update(opt_state, resharded)
```

The key insight: scalars (ndim==0) like the step counter need `NamedSharding(mesh, P())`
(fully replicated), while non-scalar leaves already get correct sharding from `reshard_to_match`.

<end>

---

## Client-Side Traceback

```
tinker.BadRequestError: Error code: 400 - {'detail': "Received incompatible devices for jitted
computation. Got argument optimizer.states[0][0] of
MaxTextBackend._create_loss_and_grad_fn.<locals>.optim_step with shape int32[] and device ids [0]
on platform TPU and jit's context mesh with device ids [0, 2, 1, 3] on platform TPU"}

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "tinker_cookbook/recipes/verifiers_rl/train.py", line 266, in <module>
    asyncio.run(cli_main(cli_config, None))
```

## Server-Side Logs (TPU)

```
POST /api/v1/asample  200 OK    (multiple sampling requests)
Batch [0:2] forward-backward time: 28.655 sec, tokens/sec: 1,143.6
(timing) process_batch_requests(process_forward_backward_batch, n=1) took 28.676s
```

forward_backward succeeded. Then optim_step fails:

```
ERROR: Error processing request 517446: Received incompatible devices for jitted computation.
  Got argument optimizer.states[0][0] of
  MaxTextBackend._create_loss_and_grad_fn.<locals>.optim_step
  with shape int32[] and device ids [0] on platform TPU
  and jit's context mesh with device ids [0, 2, 1, 3] on platform TPU

Traceback:
  engine.py:685      process_single_requests  → self.process_single_request(...)
  engine.py:661      process_single_request   → self.process_optim_step(model_id, ...)
  engine.py:513      process_optim_step       → self.backend.process_optim_step(model_id, adapter_index, request_data)
  maxtext.py:633     process_optim_step       → self._optim_step(self.model, optimizer, mean_grads)
  flax/nnx/.../compilation.py:474             → self.jitted_fn(...)
  jax/_src/pjit.py:264  cache_miss            → _python_pjit_helper(...)
  jax/_src/pjit.py:159  _python_pjit_helper   → raise ValueError(msg)

ValueError: optimizer.states[0][0] has device ids [0], jit expects mesh [0, 2, 1, 3]

(timing) process_single_request(optim_step) took 0.104s
```# Broken Resume: MaxText Backend Never Implemented Training Checkpoints

## Problem

After resuming a training run, `optim_step` fails with:
```
optimizer.states[0][0] has device ids [0], jit expects mesh [0, 2, 1, 3]
```

## The Real Root Cause: Save/Load Format Mismatch

The previous analysis (scalar sharding) was treating a symptom. The underlying problem is that
**MaxText's `save_checkpoint` never saves optimizer state** — it only exports LoRA weights in
HuggingFace PEFT format.

### Two Save Paths (disambiguation)

There are two separate save endpoints. Only the training one is broken:

| Endpoint | Engine method | Backend method | Format | Purpose |
|----------|--------------|----------------|--------|---------|
| `POST /api/v1/save_weights` | `process_save_weights` (engine.py:541) | `save_checkpoint` (maxtext.py:654) | Currently HF PEFT (**broken**) → should be flax | Training resume (optimizer + LoRA) |
| `POST /api/v1/save_weights_for_sampler` | `process_save_weights_for_sampler` (engine.py:562) | `save_sampler_checkpoint` (maxtext.py:722) | HF PEFT (**correct, unchanged**) | vLLM inference loading |

The fix only changes `save_checkpoint`. `save_sampler_checkpoint` stays HF PEFT because vLLM
needs that format for inference.

### Save Path (maxtext.py:654-669)

```python
def save_checkpoint(self, output_path, model_id, models):
    """Save training checkpoint in HuggingFace PEFT format as tar.gz."""
    with pack_and_upload(output_path) as temp_dir:
        convert_maxtext_lora_to_hf(
            lora_state=self.lora_params,
            output_path=temp_dir,
            base_model_name=self.config.base_model,
            lora_rank=self.maxtext_config.lora_rank,
            lora_alpha=self.maxtext_config.lora_alpha,
        )
```

This saves ONLY LoRA weights in HF format. **No optimizer state. No flax checkpoint format.**

### Load Path (engine.py:525-536)

```python
checkpoint = checkpoints.restore_checkpoint(  # flax.training.checkpoints
    ckpt_dir=temp_dir,
    target=self.backend.extract_checkpoint_data(model_id, self.models),  # expects {lora_weights, optimizer_state, lora_config}
    prefix="checkpoint_",
)
self.backend.insert_checkpoint_data(model_id, checkpoint, self.models)
```

This expects a **flax-format checkpoint** containing both `lora_weights` AND `optimizer_state`.

### Contrast with Native Backend (native.py:620-636)

The native backend properly implements save in flax format:
```python
def save_checkpoint(self, output_path, model_id, models):
    """Save training checkpoint as tar.gz using Flax checkpoints."""
    with pack_and_upload(output_path) as temp_dir:
        checkpoint_data = self.extract_checkpoint_data(model_id, models)  # includes optimizer!
        checkpoints.save_checkpoint(target=checkpoint_data, ckpt_dir=temp_dir, ...)
```

### Consequence

MaxText's save and load paths are **incompatible formats**:
- `save_checkpoint` → HF PEFT files (adapter_model.safetensors, adapter_config.json)
- `process_load_weights` → expects flax checkpoint with `prefix="checkpoint_"`

`flax.training.checkpoints.restore_checkpoint` cannot read HF PEFT files. It would return `None`
→ `FileNotFoundError`.

## So How Did The Error Occur?

The checkpoint at `model_9396ef01/000020.tar.gz` was either:
1. Saved by the **native backend** (which DOES save flax format + optimizer state) — implying a
   backend migration happened and now the tree structure may not match
2. Saved by an **older version** of the MaxText backend before `save_checkpoint` was changed to
   HF-only export

Either way, the checkpoint contains optimizer state from a DIFFERENT optimizer structure than what
the current MaxText backend creates. The scalar sharding error is a downstream symptom — the
optimizer state being loaded may not even be structurally compatible.

## How The Problem Was Masked

1. **In-memory cache works fine**: The eviction/re-creation path (engine.py:233-235 → 408-413)
   uses `extract_checkpoint_data` → `insert_checkpoint_data` without disk serialization. The
   optimizer state stays as live JAX arrays with correct sharding. No format mismatch ever occurs.

2. **`forward_backward` doesn't touch the optimizer**: After `insert_checkpoint_data`, the model
   and lora_params are correctly restored. `forward_backward` succeeds (as the logs show). The
   bug only surfaces when `optim_step` runs — which is the NEXT request after forward_backward.

3. **First-run training never hits the load path**: If you never resume, `register_model` creates
   a fresh optimizer with correct sharding, and everything works. The save path silently produces
   an incompatible format (HF PEFT) but nothing fails — it just writes useless files from a
   resume perspective.

4. **No error on save**: `save_checkpoint` succeeds without error — it happily writes HF PEFT
   files. There's no validation that the output is compatible with `process_load_weights`. The
   mismatch is only discovered on the next process restart when you try to resume.

5. **Error message misdirection**: The error says "device ids [0] vs mesh [0, 2, 1, 3]" which
   immediately points you toward fixing `reshard_to_match` scalar handling. You never question
   whether the checkpoint data being loaded is structurally valid in the first place.

6. **`extract_checkpoint_data` / `insert_checkpoint_data` exist → feature looks complete**: MaxText
   implements both methods with proper resharding logic. The interface contract APPEARS fulfilled.
   Nobody notices that `save_checkpoint` doesn't use `extract_checkpoint_data` — it does its own
   thing (HF PEFT export) completely disconnected from the resume machinery.

7. **The checkpoint loaded successfully → obscures its origin**: The fact that
   `model_9396ef01/000020.tar.gz` passed `restore_checkpoint` without error means it IS in flax
   format with matching tree structure. This makes it look like MaxText resume works. But the
   current MaxText `save_checkpoint` can't produce such files — the checkpoint was saved by a
   different backend or code version. The working load masks that the save is broken.

## Open Questions

1. **Which backend saved `model_9396ef01/000020.tar.gz`?** If native, the optimizer tree structure
   is different (uses `extract_adapter_state` per-adapter) vs maxtext (full `nnx.state(optimizer)`).
   `jax.tree.map` would error on structure mismatch unless they happen to align.

2. **If it WAS maxtext, when did `save_checkpoint` switch to HF-only?** The current code can't
   produce checkpoints that `process_load_weights` can consume. Resume is fundamentally broken.

3. **Should MaxText implement proper training checkpoints?** i.e. save in flax format with optimizer
   state (like native backend does), so resume actually works round-trip.

---

## Context: Resume Flow That Led to This Error

We resumed a crashed training run by reusing the same `log_path`. The flow:

1. Client-side `checkpoints.jsonl` at `/tmp/tinker-examples/rl-treebench_evalbox-30b-a3b-20260122-154327/` had:
   ```json
   {"name": "000020", "batch": 20, "state_path": "tinker://model_9396ef01/weights/000020", "sampler_path": "tinker://model_9396ef01/000020"}
   ```

2. The training code (`tinker_cookbook/rl/train.py:1302-1316`) found this checkpoint and called:
   ```python
   training_client = await service_client.create_training_client_from_state_with_optimizer_async(
       resume_info["state_path"]  # "tinker://model_9396ef01/weights/000020"
   )
   ```

3. The SDK (`.venv/.../tinker/lib/public_interfaces/service_client.py:323-342`):
   - Called `POST /api/v1/weights_info` with the old tinker path → got base_model + lora_rank
   - Called `POST /api/v1/create_model` → created NEW model `model_2596ee3b`
   - Called `POST /api/v1/load_weights` → loaded checkpoint from GCS into the new model

4. On the server (`SkyRL/skyrl-tx/tx/tinker/engine.py:515-539`), `process_load_weights`:
   - Found `/mnt/gcs_bucket/lora-experiments-checkpoints/model_9396ef01/000020.tar.gz`
   - Called `restore_checkpoint` (flax) to deserialize
   - Called `backend.insert_checkpoint_data(model_id, checkpoint, self.models)`

5. `insert_checkpoint_data` (`maxtext.py:690-720`) reshards and loads:
   ```python
   def reshard_to_match(cached, current):
       sharding = current.sharding
       return jax.device_put(cached, sharding)

   resharded_optim = jax.tree.map(
       reshard_to_match, checkpoint_data["optimizer_state"], nnx.state(optimizer)
   )
   nnx.update(nnx.state(optimizer), resharded_optim)
   ```

6. `forward_backward` succeeded. Then `optim_step` failed with the device mismatch error.

## Relevant Code Locations

- **MaxText save (HF only, no optimizer)**: `maxtext.py:654-669` (`save_checkpoint`)
- **Native save (flax format, with optimizer)**: `native.py:620-636` (`save_checkpoint`)
- **Shared load path**: `engine.py:515-536` (`process_load_weights`)
- **MaxText extract (what it WOULD save)**: `maxtext.py:671-686` (`extract_checkpoint_data`)
- **MaxText insert (what it loads into)**: `maxtext.py:690-720` (`insert_checkpoint_data`)
- **Optimizer creation**: `maxtext.py:287-293` (`register_model`)
- **optim_step JIT**: `maxtext.py:257-264` + `maxtext.py:632-633`

## Secondary Issue: Scalar Sharding (IF optimizer state is validly loaded)

Even if we fix the save format, `insert_checkpoint_data` has the scalar sharding bug. The mechanism:

1. `register_model` (line 287-293) creates the optimizer outside `jax.set_mesh` context
2. The step counter (`jnp.zeros([], jnp.int32)`) is created as an **uncommitted** scalar on device 0
3. On a fresh run this works because JAX's JIT auto-handles uncommitted arrays (can freely move
   them to the mesh)
4. But `reshard_to_match` does `jax.device_put(cached, current.sharding)` where `current.sharding`
   is `SingleDeviceSharding(device=0)` → creates a **committed** array on device [0]
5. A committed array can't be auto-moved by JIT → when `optim_step` runs under
   `jax.set_mesh(self.mesh)`, JIT sees committed device [0] vs mesh [0,2,1,3] → ValueError

The root-cause fix is wrapping `register_model` in mesh context:
```python
def register_model(self, model_id, adapter_index, lora_config):
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
    with jax.set_mesh(self.mesh):
        self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)
```

This ensures scalars get `NamedSharding(mesh, PartitionSpec())` from creation, so
`reshard_to_match` reads the correct sharding and `jax.device_put` places restored scalars on
the full mesh. No patching of `reshard_to_match` needed.

## Why Both Fixes Are Needed Together

- **Save format fix alone**: Future checkpoints save optimizer state, but on resume
  `reshard_to_match` still reads wrong sharding from the fresh optimizer → same device mismatch
- **Mesh context fix alone**: Sharding is correct on resume, but checkpoint on disk has no
  optimizer state → `restore_checkpoint` returns None → FileNotFoundError (or loads stale state)

---

## Client-Side Traceback

```
tinker.BadRequestError: Error code: 400 - {'detail': "Received incompatible devices for jitted
computation. Got argument optimizer.states[0][0] of
MaxTextBackend._create_loss_and_grad_fn.<locals>.optim_step with shape int32[] and device ids [0]
on platform TPU and jit's context mesh with device ids [0, 2, 1, 3] on platform TPU"}

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "tinker_cookbook/recipes/verifiers_rl/train.py", line 266, in <module>
    asyncio.run(cli_main(cli_config, None))
```

## Server-Side Logs (TPU)

```
POST /api/v1/asample  200 OK    (multiple sampling requests)
Batch [0:2] forward-backward time: 28.655 sec, tokens/sec: 1,143.6
(timing) process_batch_requests(process_forward_backward_batch, n=1) took 28.676s
```

forward_backward succeeded. Then optim_step fails:

```
ERROR: Error processing request 517446: Received incompatible devices for jitted computation.
  Got argument optimizer.states[0][0] of
  MaxTextBackend._create_loss_and_grad_fn.<locals>.optim_step
  with shape int32[] and device ids [0] on platform TPU
  and jit's context mesh with device ids [0, 2, 1, 3] on platform TPU

Traceback:
  engine.py:685      process_single_requests  → self.process_single_request(...)
  engine.py:661      process_single_request   → self.process_optim_step(model_id, ...)
  engine.py:513      process_optim_step       → self.backend.process_optim_step(model_id, adapter_index, request_data)
  maxtext.py:633     process_optim_step       → self._optim_step(self.model, optimizer, mean_grads)
  flax/nnx/.../compilation.py:474             → self.jitted_fn(...)
  jax/_src/pjit.py:264  cache_miss            → _python_pjit_helper(...)
  jax/_src/pjit.py:159  _python_pjit_helper   → raise ValueError(msg)

ValueError: optimizer.states[0][0] has device ids [0], jit expects mesh [0, 2, 1, 3]

(timing) process_single_request(optim_step) took 0.104s
```
# Load Checkpoint Codepath

## Current Codepath (Broken)

### 1. Client SDK triggers resume

`tinker_cookbook/rl/train.py:1302-1316`:
```python
training_client = await service_client.create_training_client_from_state_with_optimizer_async(
    resume_info["state_path"]  # "tinker://model_9396ef01/weights/000020"
)
```

### 2. SDK issues three requests

`.venv/.../tinker/lib/public_interfaces/service_client.py:323-342`:
- `POST /api/v1/weights_info` → returns base_model + lora_rank
- `POST /api/v1/create_model` → creates NEW model (e.g. `model_2596ee3b`)
- `POST /api/v1/load_weights` with `{source_model_id, checkpoint_id}`

### 3. Engine creates fresh model + optimizer

`engine.py:381-406` (`process_create_model`):
```python
self.models[model_id] = types.ModelMetadata(adapter_index=..., lora_config=..., ...)
self.backend.register_model(model_id, adapter_index, lora_config)
```

`maxtext.py:287-293` (`register_model`):
```python
def register_model(self, model_id, adapter_index, lora_config):
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
    self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)  # ← NO mesh context
```

**Problem**: Optimizer created outside `jax.set_mesh(self.mesh)`. Step counter and hyperparams
are **uncommitted** scalars on device 0 with `SingleDeviceSharding`.

### 4. Engine loads checkpoint from GCS

`engine.py:515-539` (`process_load_weights`):
```python
checkpoint_dir = self.config.checkpoints_base / request_data.source_model_id / f"{request_data.checkpoint_id}.tar.gz"

with download_and_unpack(checkpoint_dir) as temp_dir:
    checkpoint = checkpoints.restore_checkpoint(     # flax.training.checkpoints
        ckpt_dir=temp_dir,
        target=self.backend.extract_checkpoint_data(model_id, self.models),  # shape/structure reference
        prefix="checkpoint_",
    )

if checkpoint is None:
    raise FileNotFoundError(...)

self.backend.insert_checkpoint_data(model_id, checkpoint, self.models)
```

**Problem**: Expects flax-format checkpoint with `prefix="checkpoint_"`. But MaxText's
`save_checkpoint` (maxtext.py:654-669) saves HF PEFT format — incompatible. The only way this
succeeds is if the checkpoint was saved by the native backend or an older code version.

### 5. Backend reshards and inserts checkpoint data

`maxtext.py:690-720` (`insert_checkpoint_data`):
```python
optimizer = self.optimizers[model_id]

def reshard_to_match(cached, current):
    sharding = current.sharding            # ← SingleDeviceSharding(device=0) for scalars
    return jax.device_put(cached, sharding) # ← COMMITS array to device [0]

resharded_lora = jax.tree.map(reshard_to_match, checkpoint_data["lora_weights"], self.lora_params)
resharded_optim = jax.tree.map(reshard_to_match, checkpoint_data["optimizer_state"], nnx.state(optimizer))

nnx.update(self.lora_params, resharded_lora)
nnx.update(nnx.state(optimizer), resharded_optim)
self.model = nnx.merge(self.graphdef, self.lora_params, self.non_lora_params)
```

**Problem**: `current.sharding` for scalars is `SingleDeviceSharding(device=0)`. The fresh
optimizer's scalars were uncommitted (JAX JIT could freely move them), but `jax.device_put`
with explicit sharding creates a **committed** array on device [0]. JIT can no longer auto-move it.

### 6. optim_step fails

`maxtext.py:632-633` (`process_optim_step`):
```python
with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
    self._optim_step(self.model, optimizer, mean_grads)
```

JIT sees committed `optimizer.states[0][0]` on device [0] vs mesh [0,2,1,3] → ValueError.

---

## Desired Codepath (Fixed)

### Steps 1-2: Unchanged (client/SDK)

### Step 3: Create optimizer UNDER mesh context

`maxtext.py:287-293` (`register_model`):
```python
def register_model(self, model_id, adapter_index, lora_config):
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
    with jax.set_mesh(self.mesh):
        self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)
```

Now scalars get `NamedSharding(mesh, PartitionSpec())` — fully replicated across all TPU chips.

### Step 4: Unchanged (engine load path)

`process_load_weights` stays the same. It already correctly:
- Downloads and unpacks the tar.gz
- Calls `restore_checkpoint` with the right target structure
- Calls `insert_checkpoint_data`

This works because the save path (fixed separately) now produces flax-format checkpoints
with matching tree structure.

### Step 5: reshard_to_match now reads correct sharding

`maxtext.py:690-720` (`insert_checkpoint_data`) — no code change needed:
```python
def reshard_to_match(cached, current):
    sharding = current.sharding            # ← NOW: NamedSharding(mesh, P()) for scalars
    return jax.device_put(cached, sharding) # ← commits to full mesh (correct)
```

Because `register_model` created the optimizer under mesh context, `current.sharding` for
scalars is `NamedSharding(mesh, PartitionSpec())`. `jax.device_put` places the restored scalar
on all mesh devices. The committed array is now on the correct mesh.

### Step 6: optim_step succeeds

All optimizer leaves (scalars and tensors) are on the correct mesh devices. JIT sees no mismatch.

---

## Summary of Changes Required (Load Side)

| File | Line | Change |
|------|------|--------|
| `maxtext.py` | 287-293 | Wrap `nnx.Optimizer(...)` in `with jax.set_mesh(self.mesh):` |

That's it. `insert_checkpoint_data` and `process_load_weights` require no changes — they work
correctly once the optimizer is created with proper mesh context.

The load path also depends on the **save path being fixed** (separate file) to produce
flax-format checkpoints with optimizer state. Without that, `restore_checkpoint` returns None.
# Save Checkpoint Codepath: Current vs Desired

## Current Codepath (BROKEN for resume)

```
Client                              Server (API)                     Engine                              MaxText Backend
──────                              ────────────                     ──────                              ───────────────
training_client.save_state_async()
  │
  ├─► POST /api/v1/save_weights ──► api.py:779
  │                                  create_future(
  │                                    type=SAVE_WEIGHTS,
  │                                    data=SaveWeightsInput(path=name)
  │                                  )
  │                                       │
  │                                       ▼
  │                                  FutureDB (PENDING)
  │                                       │
  │                                       ▼ (TinkerEngine polls)
  │                                                                  engine.py:666
  │                                                                  process_save_weights()
  │                                                                    │  engine.py:541-560
  │                                                                    │
  │                                                                    │  output_path = config.checkpoints_base / model_id / f"{name}.tar.gz"
  │                                                                    │
  │                                                                    ▼
  │                                                                  backend.save_checkpoint(output_path, model_id, models)
  │                                                                    │  maxtext.py:654-669
  │                                                                    │
  │                                                                    ▼
  │                                                                  pack_and_upload(output_path) ──► temp_dir
  │                                                                    │  storage.py:12
  │                                                                    │
  │                                                                    ▼
  │                                                                  convert_maxtext_lora_to_hf(     ◄── PROBLEM: only saves HF PEFT
  │                                                                    lora_state=self.lora_params,       no optimizer state
  │                                                                    output_path=temp_dir,              no flax checkpoint format
  │                                                                    base_model_name=...,
  │                                                                    lora_rank=...,
  │                                                                    lora_alpha=...,
  │                                                                  )
  │                                                                    │
  │                                                                    ▼
  │                                                                  temp_dir contains:
  │                                                                    adapter_model.safetensors    ◄── HF PEFT format
  │                                                                    adapter_config.json          ◄── NOT loadable by flax checkpoints
  │                                                                    │
  │                                                                    ▼ (pack_and_upload tars + uploads)
  │                                                                  GCS: .../model_xxx/000020.tar.gz
  │
  ◄── FutureResponse(path="tinker://model_xxx/weights/000020")
```

### What gets saved to disk

```
000020.tar.gz/
├── adapter_model.safetensors    (LoRA weights in HF format)
└── adapter_config.json          (LoRA config metadata)
```

**Missing**: optimizer state (mu, nu, step counter), flax checkpoint prefix file.

---

## Desired Codepath (resume-compatible)

```
Client                              Server (API)                     Engine                              MaxText Backend
──────                              ────────────                     ──────                              ───────────────
training_client.save_state_async()
  │
  ├─► POST /api/v1/save_weights ──► api.py:779
  │                                  (same as current)
  │                                       │
  │                                       ▼
  │                                  FutureDB (PENDING)
  │                                       │
  │                                       ▼ (TinkerEngine polls)
  │                                                                  engine.py:666
  │                                                                  process_save_weights()
  │                                                                    │  engine.py:541-560
  │                                                                    │
  │                                                                    │  output_path = config.checkpoints_base / model_id / f"{name}.tar.gz"
  │                                                                    │
  │                                                                    ▼
  │                                                                  backend.save_checkpoint(output_path, model_id, models)
  │                                                                    │  maxtext.py:654-669 (CHANGED)
  │                                                                    │
  │                                                                    ▼
  │                                                                  pack_and_upload(output_path) ──► temp_dir
  │                                                                    │
  │                                                                    ▼
  │                                                                  checkpoint_data = self.extract_checkpoint_data(model_id, models)
  │                                                                    │  maxtext.py:671-688
  │                                                                    │  returns: {
  │                                                                    │    "lora_weights": jax.tree.map(jnp.copy, self.lora_params),
  │                                                                    │    "optimizer_state": jax.tree.map(jnp.copy, nnx.state(optimizer)),
  │                                                                    │    "lora_config": models[model_id].lora_config.model_dump(),
  │                                                                    │  }
  │                                                                    │
  │                                                                    ▼
  │                                                                  checkpoints.save_checkpoint(     ◄── flax format (matches load path)
  │                                                                    target=checkpoint_data,
  │                                                                    ckpt_dir=temp_dir,
  │                                                                    step=0,
  │                                                                    prefix="checkpoint_",
  │                                                                    overwrite=True,
  │                                                                  )
  │                                                                    │
  │                                                                    ▼
  │                                                                  temp_dir contains:
  │                                                                    checkpoint_0                 ◄── flax checkpoint format
  │                                                                    │
  │                                                                    ▼ (pack_and_upload tars + uploads)
  │                                                                  GCS: .../model_xxx/000020.tar.gz
  │
  ◄── FutureResponse(path="tinker://model_xxx/weights/000020")
```

### What gets saved to disk (desired)

```
000020.tar.gz/
└── checkpoint_0                 (flax msgpack checkpoint containing full dict)
    ├── lora_weights             (LoRA param arrays)
    ├── optimizer_state          (mu, nu, step counter — all with mesh sharding)
    └── lora_config              (serialized LoraConfig dict)
```

---

## Prerequisite: `register_model` mesh context (maxtext.py:287-293)

For the saved optimizer state to have correct sharding, the optimizer must be created under mesh context.

### Current (BROKEN)

```python
# maxtext.py:287-293
def register_model(self, model_id, adapter_index, lora_config):
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
    self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)
```

Step counter (`jnp.zeros([], jnp.int32)`) is created as uncommitted scalar on device 0.
`extract_checkpoint_data` then copies this scalar with `SingleDeviceSharding(device=0)`.

### Desired

```python
# maxtext.py:287-293
def register_model(self, model_id, adapter_index, lora_config):
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
    with jax.set_mesh(self.mesh):
        self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)
```

Step counter gets `NamedSharding(mesh, PartitionSpec())` (fully replicated across all devices).

---

## Reference: Native Backend (working implementation, native.py:620-636)

```python
def save_checkpoint(self, output_path, model_id, models):
    with pack_and_upload(output_path) as temp_dir:
        checkpoint_data = self.extract_checkpoint_data(model_id, models)
        checkpoints.save_checkpoint(
            target=checkpoint_data,
            ckpt_dir=temp_dir,
            step=0,
            prefix="checkpoint_",
            overwrite=True,
        )
```

---

## Files to Modify

| File | Line | Change |
|------|------|--------|
| `SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py` | 287-293 | Wrap `nnx.Optimizer(...)` in `with jax.set_mesh(self.mesh):` |
| `SkyRL/skyrl-tx/tx/tinker/backends/maxtext.py` | 654-669 | Replace `convert_maxtext_lora_to_hf(...)` with `extract_checkpoint_data` + `checkpoints.save_checkpoint` |

---

## Note on `save_sampler_checkpoint` (maxtext.py:722-737)

`save_sampler_checkpoint` also uses `convert_maxtext_lora_to_hf`. This is CORRECT — sampler
checkpoints are for inference only (loaded by vLLM/other HF-compatible tools). They do NOT need
optimizer state. Do not change this function.

---

## Note for Load Checkpoint Agent

The load path (`engine.py:515-536` → `maxtext.py:690-720`) is already correct in structure.
After the save fix, `restore_checkpoint` will find a valid flax checkpoint with optimizer state.
After the mesh context fix, `reshard_to_match` will read correct `NamedSharding(mesh, P())` from
the fresh optimizer's scalars and place restored scalars on the full mesh.

The only remaining concern on the load side: if old HF-format checkpoints exist on GCS from
before this fix, `restore_checkpoint` will return `None` → `FileNotFoundError`. These old
checkpoints are not resume-compatible and should be considered invalid.

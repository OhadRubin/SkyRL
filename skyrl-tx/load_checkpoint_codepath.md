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

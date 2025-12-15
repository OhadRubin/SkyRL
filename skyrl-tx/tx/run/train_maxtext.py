#!/usr/bin/env python3
"""Simple MaxText training test script.

This script tests MaxText model loading and forward-backward pass
with context parallelism, using nnx.Optimizer with SGD.

Usage:
    python -m tx.run.train_maxtext \
        model_name=qwen3-30b-a3b ici_context_parallelism=8 max_target_length=65536 \
        per_device_batch_size=0.125 load_parameters_path=/dev/shm/huggingface_cache/qwen3-30b-a3b-maxtext-scan/0/items
"""

import os
import time

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import optax

# MaxText imports
import MaxText
from MaxText import pyconfig as maxtext_pyconfig
from MaxText import maxtext_utils
from MaxText import model_creation_utils as maxtext_model_creation
from MaxText import sharding as maxtext_sharding
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter


def get_maxtext_base_config_path() -> str:
    """Get the absolute path to MaxText's base.yml config file."""
    maxtext_pkg_dir = os.path.dirname(MaxText.__file__)
    maxtext_root = os.path.dirname(os.path.dirname(maxtext_pkg_dir))
    config_path = os.path.join(maxtext_root, "src", "MaxText", "configs", "base.yml")
    if not os.path.exists(config_path):
        config_path = os.path.expanduser("~/maxtext/src/MaxText/configs/base.yml")
    return config_path


def parse_maxtext_config(argv: list[str]):
    """Parse MaxText config using pyconfig.initialize with argv."""
    config_path = get_maxtext_base_config_path()
    print(f"Using MaxText base config: {config_path}")
    full_argv = ["", config_path] + argv[1:]
    return maxtext_pyconfig.initialize(full_argv)


def get_maxtext_model(config, mesh):
    """Load MaxText model with Tunix adapter (like train_rl.py does)."""
    model, mesh = maxtext_model_creation.create_nnx_model(config, mesh=mesh)
    with jax.set_mesh(mesh):
        tunix_model = TunixMaxTextAdapter(base_model=model)
        tunix_model.config = None
    return tunix_model, mesh


def create_dummy_batch(batch_size: int, seq_len: int):
    """Create a dummy batch for testing."""
    input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
    target_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    loss_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    return input_ids, positions, target_ids, loss_mask


def loss_fn(model, input_ids, positions, target_ids, loss_mask):
    """Simple cross-entropy loss using TunixMaxTextAdapter call signature."""
    # TunixMaxTextAdapter signature: (input_tokens, positions, cache, attention_mask, output_hidden_states)
    logits, _ = model(input_ids, positions, None, None, False)

    logprobs = jax.nn.log_softmax(logits, axis=-1)
    target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)

    per_token_losses = -target_logprobs * loss_mask
    total_loss = per_token_losses.sum() / (loss_mask.sum() + 1e-8)

    return total_loss


def main():
    import sys

    print(f"=== MaxText Training Test ===")
    print(f"argv: {sys.argv}")
    print()

    # Parse config
    print("Parsing MaxText config...")
    config = parse_maxtext_config(sys.argv)

    # Extract key values
    num_devices = jax.device_count()
    batch_size = max(1, int(config.per_device_batch_size * num_devices))
    seq_len = config.max_target_length
    steps = getattr(config, 'steps', 3)

    print(f"  model_name: {config.model_name}")
    print(f"  ici_context_parallelism: {config.ici_context_parallelism}")
    print(f"  max_target_length: {seq_len}")
    print(f"  per_device_batch_size: {config.per_device_batch_size}")
    print(f"  num_devices: {num_devices}")
    print(f"  => batch_size: {batch_size}, seq_len: {seq_len}, steps: {steps}")
    print()

    # Create mesh
    print("Creating device mesh...")
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    print(f"  Mesh shape: {mesh.shape}")
    print(f"  Mesh axes: {mesh.axis_names}")
    print()

    # Create model with TunixMaxTextAdapter (like train_rl.py)
    print("Creating MaxText model with TunixMaxTextAdapter...")
    start = time.time()
    model, mesh = get_maxtext_model(config, mesh)
    print(f"  Model created in {time.time() - start:.1f}s")
    print()

    # Create optimizer (SGD like maxtext_deploy.py uses)
    print("Creating SGD optimizer...")
    tx = optax.sgd(learning_rate=1e-4)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    print("  Optimizer created")
    print()

    # Get data sharding
    data_sharding = maxtext_sharding.get_input_data_sharding(config, mesh)

    # Define train step
    def train_step(model, optimizer, input_ids, positions, target_ids, loss_mask):
        def loss_wrapper(model):
            return loss_fn(model, input_ids, positions, target_ids, loss_mask)

        loss, grads = nnx.value_and_grad(loss_wrapper)(model)
        optimizer.update(model, grads)
        return loss

    # JIT compile
    print("JIT compiling train step...")
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
        train_step_jit = jax.jit(
            train_step,
            in_shardings=(None, None, data_sharding, data_sharding, data_sharding, data_sharding),
        )

    # Run training steps
    print(f"\nRunning {steps} training steps...")
    for step in range(steps):
        input_ids, positions, target_ids, loss_mask = create_dummy_batch(batch_size, seq_len)

        start = time.time()
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            loss = train_step_jit(model, optimizer, input_ids, positions, target_ids, loss_mask)
        jax.block_until_ready(loss)
        elapsed = time.time() - start

        loss_val = float(loss)
        print(f"  Step {step}: loss={loss_val:.4f}, time={elapsed:.2f}s")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()

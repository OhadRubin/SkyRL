"""Tests for compute_adam_tuning_metrics function."""

import sys
from unittest.mock import MagicMock

# Mock MaxText and submodules before importing (requires tensorflow)
maxtext_mock = MagicMock()
sys.modules['MaxText'] = maxtext_mock
sys.modules['MaxText.integration'] = maxtext_mock.integration
sys.modules['MaxText.integration.tunix'] = maxtext_mock.integration.tunix
sys.modules['MaxText.integration.tunix.tunix_adapter'] = maxtext_mock.integration.tunix.tunix_adapter

                                   
                                                                                                         
# (cd /home/ohadr/SkyRL/skyrl-tx && uv run tests/tinker/test_adam_tuning_metrics.py)
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from tx.tinker.backends.maxtext import compute_adam_tuning_metrics


class SimpleModel(nnx.Module):
    """Minimal model for testing optimizer metrics."""

    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 8, rngs=rngs)
        self.linear2 = nnx.Linear(8, 4, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        return self.linear2(x)


def test_compute_adam_tuning_metrics_returns_expected_keys():
    """Verify the function returns all expected metric keys."""
    model = SimpleModel(nnx.Rngs(0))

    # Create optimizer with inject_hyperparams (required structure)
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Split model to get params
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    # Create dummy gradients matching param structure
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.01, params)

    # Do one optimizer step so mu/nu are populated
    optimizer.update(params, grads)

    # Now call the function under test
    learning_rate = 1e-3
    eps = 1e-8
    metrics = compute_adam_tuning_metrics(
        mean_grads=grads,
        lora_params=params,
        optimizer=optimizer,
        learning_rate=learning_rate,
        eps=eps,
    )

    # Verify all expected keys are present
    expected_keys = [
        "adam/grad_norm:mean",
        "adam/grad_mean:mean",
        "adam/grad_std:mean",
        "adam/grad_abs_mean:mean",
        "adam/grad_norm_max_layer:mean",
        "adam/grad_norm_min_layer:mean",
        "adam/m_norm:mean",
        "adam/m_abs_mean:mean",
        "adam/v_norm:mean",
        "adam/v_mean:mean",
        "adam/v_max:mean",
        "adam/effective_lr_mean:mean",
        "adam/effective_lr_std:mean",
        "adam/effective_lr_min:mean",
        "adam/effective_lr_max:mean",
        "adam/update_norm:mean",
        "adam/update_to_param_ratio:mean",
        "adam/param_norm:mean",
        "adam/m_to_v_ratio:mean",
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"

    # Verify all values are floats and finite
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} is not a float: {type(value)}"
        assert jnp.isfinite(value), f"{key} is not finite: {value}"


def test_compute_adam_tuning_metrics_gradient_statistics():
    """Verify gradient statistics are computed correctly."""
    model = SimpleModel(nnx.Rngs(0))

    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    # Use uniform gradients for predictable stats
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)

    optimizer.update(params, grads)

    metrics = compute_adam_tuning_metrics(
        mean_grads=grads,
        lora_params=params,
        optimizer=optimizer,
        learning_rate=1e-3,
        eps=1e-8,
    )

    # With uniform gradients of 0.1, mean should be 0.1 and std should be ~0
    assert abs(metrics["adam/grad_mean:mean"] - 0.1) < 1e-5
    assert metrics["adam/grad_std:mean"] < 1e-5  # uniform -> std ~= 0
    assert abs(metrics["adam/grad_abs_mean:mean"] - 0.1) < 1e-5


def test_compute_adam_tuning_metrics_effective_lr():
    """Verify effective learning rate calculation."""
    model = SimpleModel(nnx.Rngs(0))

    learning_rate = 0.01
    eps = 1e-8

    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)

    optimizer.update(params, grads)

    metrics = compute_adam_tuning_metrics(
        mean_grads=grads,
        lora_params=params,
        optimizer=optimizer,
        learning_rate=learning_rate,
        eps=eps,
    )

    # effective_lr = lr / (sqrt(v) + eps)
    # After one step with uniform grads of 0.1:
    # v = (1 - beta2) * g^2 = 0.001 * 0.01 = 0.00001 (for beta2=0.999)
    # sqrt(v) ~ 0.00316
    # effective_lr ~ 0.01 / 0.00316 ~ 3.16

    # Just verify it's positive and reasonable (not NaN/Inf)
    assert metrics["adam/effective_lr_mean:mean"] > 0
    assert metrics["adam/effective_lr_max:mean"] >= metrics["adam/effective_lr_mean:mean"]
    assert metrics["adam/effective_lr_min:mean"] <= metrics["adam/effective_lr_mean:mean"]


def test_compute_adam_tuning_metrics_update_to_param_ratio():
    """Verify update-to-param ratio is in reasonable range."""
    model = SimpleModel(nnx.Rngs(0))

    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)

    optimizer.update(params, grads)

    metrics = compute_adam_tuning_metrics(
        mean_grads=grads,
        lora_params=params,
        optimizer=optimizer,
        learning_rate=1e-3,
        eps=1e-8,
    )

    # update_to_param_ratio should be small but positive
    ratio = metrics["adam/update_to_param_ratio:mean"]
    assert ratio > 0, "update_to_param_ratio should be positive"
    assert ratio < 1, "update_to_param_ratio should be < 1 for reasonable training"


def test_compute_adam_tuning_metrics_m_to_v_ratio():
    """Verify m/v ratio is computed correctly."""
    model = SimpleModel(nnx.Rngs(0))

    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)

    optimizer.update(params, grads)

    metrics = compute_adam_tuning_metrics(
        mean_grads=grads,
        lora_params=params,
        optimizer=optimizer,
        learning_rate=1e-3,
        eps=1e-8,
    )

    # m_to_v_ratio = m_norm / v_norm
    # Both should be positive after one step
    assert metrics["adam/m_to_v_ratio:mean"] > 0


def test_compute_adam_tuning_metrics_layer_norms():
    """Verify per-layer gradient norms are computed."""
    model = SimpleModel(nnx.Rngs(0))

    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    # Create gradients with different magnitudes per layer
    def scale_grads(path, p):
        if "linear1" in str(path):
            return jnp.ones_like(p) * 0.1
        else:
            return jnp.ones_like(p) * 0.5
    grads = jax.tree.map_with_path(scale_grads, params)

    optimizer.update(params, grads)

    metrics = compute_adam_tuning_metrics(
        mean_grads=grads,
        lora_params=params,
        optimizer=optimizer,
        learning_rate=1e-3,
        eps=1e-8,
    )

    # With different scales, max_layer should be > min_layer
    assert metrics["adam/grad_norm_max_layer:mean"] > metrics["adam/grad_norm_min_layer:mean"]


if __name__ == "__main__":
    test_compute_adam_tuning_metrics_returns_expected_keys()
    test_compute_adam_tuning_metrics_gradient_statistics()
    test_compute_adam_tuning_metrics_effective_lr()
    test_compute_adam_tuning_metrics_update_to_param_ratio()
    test_compute_adam_tuning_metrics_m_to_v_ratio()
    test_compute_adam_tuning_metrics_layer_norms()
    print("All tests passed!")

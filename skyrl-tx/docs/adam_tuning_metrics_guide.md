# Adam Optimizer Tuning Metrics Guide

This guide explains the metrics logged by `compute_adam_tuning_metrics()` in `tx/tinker/backends/maxtext.py` and how to use them for tuning Adam hyperparameters: `beta1`, `beta2`, and `eps`.

## Quick Reference

| Hyperparameter | Default | What It Controls | Key Diagnostic Metrics |
|----------------|---------|------------------|------------------------|
| `beta1` | 0.9 | Momentum decay (first moment) | `grad_std`, `m_norm`, `m_to_v_ratio` |
| `beta2` | 0.95 | Adaptive LR decay (second moment) | `v_norm`, `v_max`, `effective_lr_*` |
| `eps` | 1e-8 | Numerical stability | `effective_lr_std`, `effective_lr_min` |

---

## Metric Categories

### 1. Gradient Statistics (`adam/grad_*`)

These metrics describe the raw gradients before Adam processes them.

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `grad_norm` | L2 norm of all gradients | 0.01 - 10.0 |
| `grad_mean` | Mean gradient value | Near 0 (centered) |
| `grad_std` | Standard deviation of gradients | 0.001 - 1.0 |
| `grad_abs_mean` | Mean of absolute gradient values | 0.001 - 0.1 |
| `grad_norm_max_layer` | Largest per-layer gradient norm | < 10x `grad_norm_min_layer` |
| `grad_norm_min_layer` | Smallest per-layer gradient norm | > 0 |

**Interpretation:**

- **`grad_norm` exploding (> 100):** Gradients are too large. Consider gradient clipping or reducing learning rate.

- **`grad_norm` vanishing (< 1e-6):** Gradients are too small. Model may have converged or learning rate is too low.

- **`grad_std` very high relative to `grad_abs_mean`:** High gradient variance. This suggests:
  - Increase `beta1` to add more momentum smoothing
  - Or reduce batch size variation

- **`grad_norm_max_layer` >> `grad_norm_min_layer` (10x+ ratio):** Layer imbalance. Some layers are learning much faster than others. Consider per-layer learning rates or better initialization.

---

### 2. First Moment Statistics (`adam/m_*`)

The first moment `m` is Adam's exponential moving average of gradients (momentum).

| Metric | Description | What It Indicates |
|--------|-------------|-------------------|
| `m_norm` | L2 norm of momentum buffer | Accumulated gradient direction |
| `m_abs_mean` | Mean absolute momentum value | Average momentum magnitude |

**Interpretation:**

- **`m_norm` growing unboundedly:** Gradients are consistently pointing in the same direction. This is normal early in training but should stabilize.

- **`m_norm` oscillating wildly:** Gradient direction is unstable. Consider:
  - Increase `beta1` (e.g., 0.9 -> 0.95) for more smoothing
  - Or the loss landscape is very noisy

- **`m_norm` near zero despite nonzero gradients:** `beta1` may be too low, causing momentum to decay too quickly.

**Tuning `beta1`:**

```
beta1 controls how much past gradients influence the current update.

Higher beta1 (0.95-0.99):
  - More momentum, smoother updates
  - Better for noisy gradients (high grad_std)
  - Slower to change direction

Lower beta1 (0.8-0.9):
  - Less momentum, more responsive
  - Better for stable gradients
  - Faster adaptation to gradient changes
```

---

### 3. Second Moment Statistics (`adam/v_*`)

The second moment `v` is Adam's exponential moving average of squared gradients. It provides per-parameter adaptive learning rates.

| Metric | Description | What It Indicates |
|--------|-------------|-------------------|
| `v_norm` | L2 norm of second moment buffer | Overall adaptive scaling |
| `v_mean` | Mean second moment value | Average squared gradient history |
| `v_max` | Maximum second moment value | Largest adaptive denominator |

**Interpretation:**

- **`v_norm` growing unboundedly:** Squared gradients are accumulating. This causes effective learning rate to shrink over time (Adam's "learning rate decay" effect).

- **`v_max` >> `v_mean` (100x+ ratio):** Some parameters have seen much larger gradients than others. This creates very different effective learning rates across parameters.

- **`v_mean` very small (< 1e-12):** Second moment hasn't accumulated much. Either training just started or gradients are extremely small.

**Tuning `beta2`:**

```
beta2 controls how quickly the adaptive learning rate adjusts.

Higher beta2 (0.99-0.999):
  - Slower adaptation
  - More stable effective learning rates
  - Better for consistent gradient magnitudes

Lower beta2 (0.9-0.95):
  - Faster adaptation
  - Effective LR responds quickly to gradient changes
  - Better for non-stationary problems
  - Risk: Can cause oscillation if gradients are noisy
```

---

### 4. Effective Learning Rate (`adam/effective_lr_*`)

The effective learning rate is what Adam actually applies: `lr / (sqrt(v) + eps)`.

| Metric | Description | What It Indicates |
|--------|-------------|-------------------|
| `effective_lr_mean` | Mean effective LR across params | Average actual step size |
| `effective_lr_std` | Std dev of effective LR | Spread of step sizes |
| `effective_lr_min` | Minimum effective LR | Slowest-learning parameter |
| `effective_lr_max` | Maximum effective LR | Fastest-learning parameter |

**Interpretation:**

- **`effective_lr_mean` << nominal `lr`:** Second moment `v` has grown large, shrinking effective learning rates. This is expected late in training.

- **`effective_lr_std` very high:** Parameters have very different effective learning rates. Some are learning much faster than others.

- **`effective_lr_min` near zero:** Some parameters have effectively stopped learning. Their `v` values are very large.

- **`effective_lr_max` >> `effective_lr_mean`:** Some parameters have very small `v` values (haven't seen large gradients). They may be updating too aggressively.

**Tuning `eps`:**

```
eps prevents division by zero when v is small.

Higher eps (1e-6 to 1e-4):
  - Caps effective_lr_max
  - More uniform learning rates across parameters
  - Use when effective_lr_std is very high
  - Use when seeing NaN/Inf

Lower eps (1e-10 to 1e-8):
  - Allows more per-parameter adaptation
  - Better for well-behaved gradients
  - Risk: Division instability if v is tiny
```

**Key diagnostic:** If `effective_lr_std / effective_lr_mean > 10`, consider increasing `eps`.

---

### 5. Update Statistics (`adam/update_*`, `adam/param_*`)

These metrics describe the actual parameter updates.

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `update_norm` | L2 norm of parameter update | 0.0001 - 0.1 |
| `param_norm` | L2 norm of parameters | Model-dependent |
| `update_to_param_ratio` | `update_norm / param_norm` | 1e-5 to 1e-2 |

**Interpretation:**

- **`update_to_param_ratio` > 0.1:** Updates are very large relative to parameters. Risk of instability. Reduce learning rate.

- **`update_to_param_ratio` < 1e-6:** Updates are tiny. Learning may be too slow. Increase learning rate or check for vanishing gradients.

- **`update_norm` exploding:** Immediate instability risk. Reduce learning rate or add gradient clipping.

---

### 6. Balance Metric (`adam/m_to_v_ratio`)

| Metric | Description | What It Indicates |
|--------|-------------|-------------------|
| `m_to_v_ratio` | `m_norm / v_norm` | Balance between momentum and adaptation |

**Interpretation:**

- **High ratio (> 10):** Momentum dominates. Updates are mostly determined by accumulated gradient direction, less by per-parameter scaling.

- **Low ratio (< 0.1):** Adaptive scaling dominates. Updates are mostly determined by per-parameter learning rates.

- **Balanced (0.1 - 10):** Both momentum and adaptation contribute meaningfully.

---

## Common Tuning Scenarios

### Scenario 1: Training is unstable (loss spikes)

**Symptoms:**
- `grad_norm` spikes
- `update_to_param_ratio` > 0.1
- Loss oscillates or diverges

**Solutions:**
1. Reduce learning rate
2. Increase `beta1` (more momentum smoothing)
3. Add gradient clipping

---

### Scenario 2: Training is too slow

**Symptoms:**
- `update_to_param_ratio` < 1e-6
- `effective_lr_mean` << nominal lr
- `v_norm` is very large

**Solutions:**
1. Increase learning rate
2. Decrease `beta2` (faster v adaptation, less LR decay)
3. Reset optimizer state (if v has accumulated too much history)

---

### Scenario 3: Some parameters not learning

**Symptoms:**
- `effective_lr_min` << `effective_lr_mean` (100x+ difference)
- `effective_lr_std` very high
- `grad_norm_max_layer` >> `grad_norm_min_layer`

**Solutions:**
1. Increase `eps` to cap effective LR spread
2. Check for dead layers (zero gradients)
3. Consider per-layer learning rates

---

### Scenario 4: Noisy/oscillating loss

**Symptoms:**
- `grad_std` >> `grad_abs_mean`
- `m_norm` oscillates
- Loss zigzags without clear trend

**Solutions:**
1. Increase `beta1` (more momentum smoothing)
2. Increase batch size (reduce gradient noise)
3. Increase `beta2` (more stable effective LR)

---

## Recommended Starting Points

| Setting | Stable Training | Aggressive Training |
|---------|-----------------|---------------------|
| `beta1` | 0.9 | 0.85 |
| `beta2` | 0.999 | 0.95 |
| `eps` | 1e-8 | 1e-8 |

For RL training (which tends to be noisy):
- `beta1=0.9` - standard momentum
- `beta2=0.95` - faster adaptation than default (0.999)
- `eps=1e-8` - standard

---

## Logging Integration

These metrics are logged via `ml_logger.log_metrics()` and appear in wandb under the `adam/` prefix:

```python
# Example wandb query
wandb.Api().runs("project")[0].history(keys=[
    "adam/grad_norm",
    "adam/effective_lr_mean",
    "adam/update_to_param_ratio",
])
```

All metrics use the `:mean` suffix convention for proper aggregation across batches.

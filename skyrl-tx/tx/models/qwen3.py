import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List

from flax import nnx
from flax.nnx.nn.lora import LoRALinear, LoRAParam
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec as P
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

from tx.layers.lora import LoRAEmbed
from tx.layers.util import Param, prepare_routing
from tx.models.configs import Qwen3Config
from tx.models.experts import fused_moe_func
from tx.models.types import CausalLMOutput, ModelOutput
from tx.utils.generator import GeneratorMixin, KVCache, compute_positions
import flax.linen as nn

# Import maxtext's RoutedMoE
from MaxText.layers.moe import RoutedMoE
from MaxText.common_types import DecoderBlockType, ShardMode


def _skyrl_logical_axis_rules():
    """Logical axis rules mapping MaxText names to SkyRL mesh axes (layer, dp, tensor)."""
    return (
        # Batch/data axes -> dp
        ('activation_batch', ('dp',)),
        ('activation_batch_no_exp', ('dp',)),
        ('activation_embed_and_logits_batch', ('dp',)),
        ('activation_embed_and_logits_batch_sequence', ('dp',)),
        ('activation_prefill_kv_batch', ('dp',)),
        ('activation_kv_batch', ('dp',)),
        ('activation_kv_batch_no_exp', ('dp',)),
        ('decode_batch', ('dp',)),
        # Head axes -> tensor
        ('activation_heads', ('tensor',)),
        ('activation_kv_heads', ('tensor',)),
        # Length/sequence axes -> None (not sharded)
        ('activation_length', ()),
        ('activation_length_no_exp', ()),
        ('activation_norm_length', ()),
        ('activation_q_length', ()),
        ('activation_q_length_no_exp', ()),
        ('prefill_activation_length', ()),
        ('prefill_activation_norm_length', ()),
        ('activation_kv_length', ()),
        ('decode_length', ()),
        # Embedding/MLP axes -> tensor for tensor parallelism
        ('activation_embed', ('tensor',)),
        ('activation_mlp', ('tensor',)),
        ('activation_kv', ('tensor',)),
        ('activation_kv_head_dim', ('tensor',)),
        ('activation_vocab', ('tensor',)),
        ('activation_exp', ()),  # No expert parallelism
        ('activation_stage', ()),
        # Weight axes
        ('mlp', ('tensor',)),
        ('mlp_no_fsdp', ('tensor',)),
        ('vocab', ('tensor',)),
        ('heads', ('tensor',)),
        ('q_heads', ('tensor',)),
        ('kv_heads', ('tensor',)),
        ('embed', ()),  # Usually not sharded for weights
        ('embed_no_exp', ()),
        ('embed_tensor_transpose', ()),
        ('exp', ()),  # Expert axis - no expert parallelism
        ('q_lora', ()),
        ('kv_lora', ()),
        # Cache axes
        ('cache_batch', ('dp',)),
        ('cache_batch_prefill', ('dp',)),
        ('cache_sequence', ()),
        ('cache_heads', ('tensor',)),
        ('cache_heads_none', ()),
        ('cache_kv', ('tensor',)),
        ('cache_scale_batch', ('dp',)),
        ('cache_scale_sequence', ()),
        ('cache_scale_heads', ('tensor',)),
        ('cache_scale_kv', ('tensor',)),
    )


@dataclass
class MaxTextConfigAdapter:
    """Config wrapper to use MaxText's RoutedMoE with Qwen3Config.

    Maps maxtext YAML config fields to the attributes RoutedMoE expects via self.config.X
    """

    # Core dimensions (from YAML)
    emb_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_mlp_dim: int  # base_moe_mlp_dim in YAML

    # Model architecture (from YAML)
    num_decoder_layers: int = 48  # base_num_decoder_layers
    num_query_heads: int = 32  # base_num_query_heads
    num_kv_heads: int = 4  # base_num_kv_heads
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6  # normalization_layer_epsilon
    use_qk_norm: bool = True
    rope_theta: int = 10_000_000  # rope_max_timescale

    # Model identification
    model_name: str = "qwen3_moe"
    decoder_block: DecoderBlockType = DecoderBlockType.QWEN3_MOE

    # Routing config (Qwen3 defaults)
    routed_bias: bool = False
    routed_score_func: str = "softmax"
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True  # Qwen3-specific
    n_routing_groups: int = -1  # No routing groups
    topk_routing_group: int = 1
    use_random_routing: bool = False

    # Computation config
    dtype: Any = jnp.bfloat16
    weight_dtype: Any = jnp.bfloat16
    matmul_precision: str = "default"
    mlp_activations: List[str] = field(default_factory=lambda: ["silu", "linear"])
    mlp_bias: bool = False
    float32_weight_sum: bool = False
    activations_in_float32: bool = False
    enable_dropout: bool = False
    dropout_rate: float = 0.0

    # Sharding config
    shard_mode: ShardMode = ShardMode.AUTO
    fsdp_shard_on_exp: bool = False
    attention: str = "dot_product"  # Not vllm_rpa
    logical_axis_rules: Any = field(default_factory=_skyrl_logical_axis_rules)

    # Backend config
    sparse_matmul: bool = True  # Use GMM-based sparse matmul
    megablox: bool = True
    use_tokamax_gmm: bool = False
    quantization: Any = None
    use_qwix_quantization: bool = False

    # Tiling config (defaults from maxtext)
    wi_tile_fwd_batch_seq: int = 512
    wi_tile_fwd_embed_dim: int = 0
    wi_tile_fwd_mlp_dim: int = 0
    wi_tile_dlhs_batch_seq: int = 0
    wi_tile_dlhs_embed_dim: int = 0
    wi_tile_dlhs_mlp_dim: int = 0
    wi_tile_drhs_batch_seq: int = 0
    wi_tile_drhs_embed_dim: int = 0
    wi_tile_drhs_mlp_dim: int = 0
    wo_tile_fwd_batch_seq: int = 0
    wo_tile_fwd_embed_dim: int = 0
    wo_tile_fwd_mlp_dim: int = 0
    wo_tile_dlhs_batch_seq: int = 0
    wo_tile_dlhs_embed_dim: int = 0
    wo_tile_dlhs_mlp_dim: int = 0
    wo_tile_drhs_batch_seq: int = 0
    wo_tile_drhs_embed_dim: int = 0
    wo_tile_drhs_mlp_dim: int = 0

    # Advanced options
    use_ring_of_experts: bool = False
    use_custom_sort_vjp: bool = True
    mlp_activations_limit: float = 10.0
    capacity_factor: float = 0.0  # 0 = no capacity limit
    load_balance_loss_weight: float = 0.01
    model_call_mode: str = "train"  # "train" or "inference"
    moe_fsdp_use_two_stage_all_gather: bool = False
    shared_experts: int = 0  # For DeepSeek-style shared experts

    @classmethod
    def from_qwen3_config(cls, config: Qwen3Config, dtype=jnp.bfloat16) -> "MaxTextConfigAdapter":
        """Create MaxText config from Qwen3Config."""
        return cls(
            emb_dim=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_mlp_dim=config.moe_intermediate_size,
            num_decoder_layers=config.num_hidden_layers,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            dtype=dtype,
            weight_dtype=dtype,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, dtype=jnp.bfloat16) -> "MaxTextConfigAdapter":
        """Create MaxText config from a maxtext YAML config file.

        Args:
            yaml_path: Path to maxtext config file (e.g., ~/maxtext/src/MaxText/configs/models/qwen3-30b-a3b.yml)
            dtype: JAX dtype for the model
        """
        import os
        import yaml

        yaml_path = os.path.expanduser(yaml_path)
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Map maxtext YAML fields to MaxTextConfigAdapter fields
        return cls(
            # Core dimensions
            emb_dim=cfg.get('base_emb_dim', 2048),
            num_experts=cfg.get('num_experts', 128),
            num_experts_per_tok=cfg.get('num_experts_per_tok', 8),
            moe_mlp_dim=cfg.get('base_moe_mlp_dim', 768),
            # Model architecture
            num_decoder_layers=cfg.get('base_num_decoder_layers', 48),
            num_query_heads=cfg.get('base_num_query_heads', 32),
            num_kv_heads=cfg.get('base_num_kv_heads', 4),
            head_dim=cfg.get('head_dim', 128),
            vocab_size=cfg.get('vocab_size', 151936),
            rms_norm_eps=cfg.get('normalization_layer_epsilon', 1e-6),
            use_qk_norm=cfg.get('use_qk_norm', True),
            rope_theta=cfg.get('rope_max_timescale', 10_000_000),
            # Computation
            mlp_activations=cfg.get('mlp_activations', ['silu', 'linear']),
            norm_topk_prob=cfg.get('norm_topk_prob', True),
            enable_dropout=cfg.get('enable_dropout', False),
            dtype=dtype,
            weight_dtype=dtype,
        )

    @classmethod
    def qwen3_30b_a3b(cls, dtype=jnp.bfloat16) -> "MaxTextConfigAdapter":
        """Create MaxText config for Qwen3-30B-A3B from ~/maxtext config."""
        return cls.from_yaml('~/maxtext/src/MaxText/configs/models/qwen3-30b-a3b.yml', dtype=dtype)

class RMSNorm(nnx.Module):
    def __init__(self, size: int, *, eps: float = 1e-6, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(
            size, dtype=dtype, kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding_names=(None,)), rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms


def apply_rope(inputs: jax.Array, position_ids: jax.Array, head_dim: int, theta: int) -> jax.Array:
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = jnp.pow(theta, fraction)
    x = (position_ids[..., None] / timescale[None, None, :])[..., None, :]
    sin, cos = jnp.sin(x), jnp.cos(x)
    a, b = jnp.split(inputs, 2, axis=-1)
    return jnp.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(inputs.dtype)


class Qwen3Attention(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs, mesh=None) -> None:
        self.config = config
        self.mesh = mesh
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        tp = get_abstract_mesh().shape.get("tensor", 1)
        shard_attention_heads = config.shard_attention_heads
        shard_attention_heads = False
        if shard_attention_heads:
            assert self.num_heads % tp == 0, f"num_heads={self.num_heads} must be divisible by tp={tp}"
            # assert self.num_kv_heads % tp == 0, f"num_kv_heads={self.num_kv_heads} must be divisible by tp={tp}"
        tp_shard = "tensor" if shard_attention_heads else None
        self.tp_shard = tp_shard
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads

        lora_rank = config.lora_rank if getattr(config, "attn_lora", True) else 0
        self.q_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            lora_rank=lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, tp_shard)),
            rngs=rngs,
        )
        self.k_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            lora_rank=lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, tp_shard)),
            rngs=rngs,
        )
        self.v_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            lora_rank=lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, tp_shard)),
            rngs=rngs,
        )
        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.head_dim,
            out_features=config.hidden_size,
            lora_rank=lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(tp_shard, None)),
            rngs=rngs,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        # Remat the attention computation to save memory during backward pass
        # This is applied here (inside the layer) rather than around the whole layer
        # to ensure scan carry flows correctly without 48x broadcast buffers
        @nnx.remat(policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
        def _attention_forward(self, x, attention_mask, positions, kv_cache):
            return self._forward(x, attention_mask=attention_mask, positions=positions, kv_cache=kv_cache)
        return _attention_forward(self, x, attention_mask, positions, kv_cache)

    def _forward(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        B, T, _ = x.shape

        # Project and reshape to [B, T, num_heads, head_dim]
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, positions, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, positions, self.head_dim, self.config.rope_theta)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache, cache_position = kv_cache
            k = jax.lax.dynamic_update_slice(k_cache, k, (0, cache_position, 0, 0))
            v = jax.lax.dynamic_update_slice(v_cache, v, (0, cache_position, 0, 0))

        updated_cache = (k, v)

        # Check if we should use ring attention
        use_ring = (
            getattr(self.config, "use_ring_attention", False)
            and self.mesh is not None
            and kv_cache is None  # Ring attention only for training, not inference
        )
        use_ring = False
        use_flash = True

        if use_ring:
            # Ring attention path - uses [B, T, num_heads, head_dim] layout
            from ringattention import ringattention

            # Handle GQA: repeat k/v heads to match num_heads (in BTHD layout)
            if self.num_kv_heads < self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = jnp.repeat(k, n_rep, axis=2)  # [B, T, num_heads, head_dim]
                v = jnp.repeat(v, n_rep, axis=2)  # [B, T, num_heads, head_dim]

            # Transform attention mask to bias: [B, 1, 1, T] -> float mask
            attention_mask_expanded = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = jax.lax.select(
                attention_mask_expanded > 0,
                jnp.zeros_like(attention_mask_expanded, dtype=x.dtype),
                jnp.full_like(attention_mask_expanded, jnp.finfo(x.dtype).min, dtype=x.dtype),
            )

            # Wrap ringattention in nnx.shard_map - use "tensor" axis for sequence parallelism
            # Ring attention distributes sequence across devices in a ring topology
            ring_attention_sharded = nnx.shard_map(
                partial(
                    ringattention,
                    axis_name="tensor",  # Use tp axis for ring communication
                    float32_logits=True,
                    cache_idx=None,
                    blockwise_kwargs=dict(
                        causal_block_size=1,
                        deterministic=True,
                        dropout_rng=None,
                        attn_pdrop=0.0,
                        query_chunk_size=self.config.scan_query_chunk_size,
                        key_chunk_size=self.config.scan_key_chunk_size,
                        dtype=x.dtype,
                        policy=jax.checkpoint_policies.nothing_saveable,
                        precision=None,
                        prevent_cse=not getattr(self.config, "scan_layers", False),
                    ),
                ),
                mesh=self.mesh,
                in_specs=(
                    P("dp", "tensor", None, None),  # q: [B, T, num_heads, head_dim]
                    P("dp", "tensor", None, None),  # k: [B, T, num_heads, head_dim]
                    P("dp", "tensor", None, None),  # v: [B, T, num_heads, head_dim]
                    P("dp", None, None, None),  # bias: [B, 1, 1, T]
                    P("dp", None),              # segment_ids (None)
                ),
                out_specs=P("dp", "tensor", None, None),
            )
            attn_output = ring_attention_sharded(q, k, v, attention_bias, None)
            # attn_output: [B, T, num_heads, head_dim]
        elif use_flash:
            # Standard flash attention path
            # Transpose to [B, num_heads, T, head_dim] for flash attention
            q = jnp.transpose(q, (0, 2, 1, 3))  # [B, num_heads, T, head_dim]
            k = jnp.transpose(k, (0, 2, 1, 3))  # [B, num_kv_heads, T, head_dim]
            v = jnp.transpose(v, (0, 2, 1, 3))  # [B, num_kv_heads, T, head_dim]

            # Handle GQA: repeat k/v heads to match num_heads
            if self.num_kv_heads < self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = jnp.repeat(k, n_rep, axis=1)  # [B, num_heads, T, head_dim]
                v = jnp.repeat(v, n_rep, axis=1)  # [B, num_heads, T, head_dim]
            k = jax.lax.with_sharding_constraint(k, P(("dp","layer"),"tensor", None, None) )
            v = jax.lax.with_sharding_constraint(v, P(("dp","layer"),"tensor", None, None))

            # Use Pallas flash attention wrapped in shard_map
            # Mosaic kernels cannot be automatically partitioned, so we must use shard_map
            # Layout: [batch, num_heads, seq_len, head_dim]
            # Sharding: batch on "dp", heads on "tensor" (if shard_attention_heads), seq/head_dim unsharded
            sm_scale = 1.0 / math.sqrt(self.head_dim)
            causal = kv_cache is None

            if self.mesh is not None:
                # Wrap flash_attention in nnx.shard_map for proper SPMD partitioning
                attn_spec = P(("dp","layer"), self.tp_shard, None, None)
                flash_attention_sharded = nnx.shard_map(
                    partial(flash_attention, causal=causal, sm_scale=sm_scale),
                    mesh=self.mesh,
                    in_specs=(attn_spec, attn_spec, attn_spec),
                    out_specs=attn_spec,
                    check_vma=False,
                )
                attn_output = flash_attention_sharded(q, k, v)
            else:
                # Fallback without shard_map (e.g., single device)
                attn_output = flash_attention(q, k, v, causal=causal, sm_scale=sm_scale)
            # attn_output: [B, num_heads, T, head_dim]

            # Transpose back to [B, T, num_heads, head_dim]
            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [B, T, num_heads, head_dim]
        else:
            attn_output = jax.nn.dot_product_attention(
                q,
                k,
                v,
                scale=1.0 / self.head_dim**0.5,
                mask=attention_mask[:, None, None, :].astype(bool),
                is_causal=kv_cache is None,
            )

        output = attn_output.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(output), updated_cache


class Qwen3MLP(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        lora_rank = config.lora_rank if getattr(config, "mlp_lora", True) else 0
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, "tensor")),
            lora_rank=lora_rank,
            rngs=rngs,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, "tensor")),
            lora_rank=lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=("tensor", None)),
            lora_rank=lora_rank,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Remat the MLP computation to save memory during backward pass
        # This is applied here (inside the layer) rather than around the whole layer
        # to ensure scan carry flows correctly without 48x broadcast buffers
        @nnx.remat(policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
        def _mlp_forward(self, x):
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)
            return self.down_proj(nnx.silu(gate_out) * up_out)
        return _mlp_forward(self, x)


class Qwen3Experts(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        use_fused_moe = getattr(config, 'use_fused_moe', False)

        if use_fused_moe:
            # Fused MoE format: w1 combines gate+up, w2 is transposed down
            # w1: (E, I*2, H) - gate and up merged
            # w2: (E, H, I) - down transposed
            self.w1 = nnx.Param(
                nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, "tensor", None))(
                    rngs.params(), (config.num_experts, config.moe_intermediate_size * 2, config.hidden_size), dtype
                )
            )
            self.w2 = nnx.Param(
                nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, None, "tensor"))(
                    rngs.params(), (config.num_experts, config.hidden_size, config.moe_intermediate_size), dtype
                )
            )
        else:
            # Standard format for ragged_dot
            # gate_proj/up_proj: (E, H, I)
            # down_proj: (E, I, H)
            self.gate_proj = nnx.Param(
                nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, None, "tensor"))(
                    rngs.params(), (config.num_experts, config.hidden_size, config.moe_intermediate_size), dtype
                )
            )
            self.up_proj = nnx.Param(
                nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, None, "tensor"))(
                    rngs.params(), (config.num_experts, config.hidden_size, config.moe_intermediate_size), dtype
                )
            )
            self.down_proj = nnx.Param(
                nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, "tensor", None))(
                    rngs.params(), (config.num_experts, config.moe_intermediate_size, config.hidden_size), dtype
                )
            )

    def __call__(self, hidden_states: jax.Array, router_logits: jax.Array, mesh=None) -> jax.Array:
        if getattr(self.config, 'use_fused_moe', False):
            # Capture w1/w2 in closure (saved), only checkpoint hidden_states/router_logits
            w1, w2 = self.w1.value, self.w2.value
            @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
            def _fused_moe_forward(hidden_states, router_logits):
                return fused_moe_func(
                    hidden_states=hidden_states,
                    w1=w1,
                    w2=w2,
                    w1_bias=None,
                    w2_bias=None,
                    gating_output=router_logits,
                    topk=self.config.num_experts_per_tok,
                    renormalize=True,
                    mesh=mesh,
                    use_ep=False,
                    activation="silu",
                )
            return _fused_moe_forward(hidden_states, router_logits)

        # Remat the MoE expert computation to save memory during backward pass (ragged_dot path only)
        @nnx.remat(policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
        def _experts_forward(self, hidden_states, router_logits):
            # Get top-k experts for each token and compute routing weights
            routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
            routing_weights = nnx.softmax(routing_weights, axis=-1)

            # Prepare for ragged_dot by sorting tokens based on their assigned expert
            selected_experts_flat = selected_experts.ravel()
            hidden_states_expanded = jnp.repeat(hidden_states, self.config.num_experts_per_tok, axis=0)
            hidden_states_sorted, group_sizes, unsort_indices, _ = prepare_routing(
                hidden_states_expanded,
                selected_experts_flat,
                self.config.num_experts,
            )

            # Apply expert layers using ragged_dot
            gate_out = jax.lax.ragged_dot(hidden_states_sorted, self.gate_proj.value, group_sizes)
            up_out = jax.lax.ragged_dot(hidden_states_sorted, self.up_proj.value, group_sizes)
            down_out = jax.lax.ragged_dot(nnx.silu(gate_out) * up_out, self.down_proj.value, group_sizes)

            # Unsort and combine the expert outputs
            unsorted_out = down_out[unsort_indices]
            reshaped_out = unsorted_out.reshape(-1, self.config.num_experts_per_tok, self.config.hidden_size)
            return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)
        return _experts_forward(self, hidden_states, router_logits)


class Qwen3MoeSparseMoeBlock(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs, mesh=None) -> None:
        self.config = config
        self.mesh = mesh
        self.use_maxtext_moe = getattr(config, 'use_maxtext_moe', False)

        if self.use_maxtext_moe:
            # Use MaxText's RoutedMoE
            from MaxText.layers.initializers import nd_dense_init
            maxtext_config = MaxTextConfigAdapter.from_qwen3_config(config, dtype=dtype)
            self.moe_block = RoutedMoE(
                config=maxtext_config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                mesh=mesh,
                kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
                kernel_axes=("embed", None),
                intermediate_dim=config.moe_intermediate_size,
                dtype=dtype,
                weight_dtype=dtype,
                quant=None,
                rngs=rngs,
            )
        else:
            # Original implementation
            self.gate = nnx.Linear(
                config.hidden_size,
                config.num_experts,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, None)),
                rngs=rngs,
            )
            self.experts = Qwen3Experts(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        return_router_logits: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        if self.use_maxtext_moe:
            # MaxText's RoutedMoE handles everything
            output, load_balance_loss = self.moe_block(hidden_states)
            if return_router_logits:
                # RoutedMoE doesn't return router_logits directly, return None
                return output, None
            return output
        else:
            # Original implementation
            (batch_size, seq_len, hidden_size) = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_size)
            router_logits = self.gate(hidden_states)

            hidden_states = self.experts(hidden_states, router_logits, mesh=self.mesh)
            hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)

            if return_router_logits:
                return hidden_states, router_logits
            return hidden_states


class Qwen3DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs, mesh=None) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Qwen3Attention(config, dtype=dtype, rngs=rngs, mesh=mesh)
        if getattr(config, "num_experts", None):
            self.mlp = Qwen3MoeSparseMoeBlock(config, dtype=dtype, rngs=rngs, mesh=mesh)
        else:
            self.mlp = Qwen3MLP(config, dtype=dtype, rngs=rngs)
    @nnx.remat(policy=jax.checkpoint_policies.nothing_saveable)
    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        positions: jax.Array,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, updated_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            positions=positions,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, updated_cache


class Qwen3Model(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs, mesh=None) -> None:
        self.config = config

        embed_lora = getattr(config, "embed_lora", True)
        if embed_lora and config.lora_rank > 0:
            self.embed_tokens = LoRAEmbed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=dtype,
                lora_rank=config.lora_rank,
                param_dtype=dtype,
                embedding_init=nnx.with_metadata(nnx.initializers.normal(), sharding_names=("tensor", None)),
                rngs=rngs,
            )
        else:
            self.embed_tokens = nnx.Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                embedding_init=nnx.with_metadata(nnx.initializers.normal(), sharding_names=("tensor", None)),
                rngs=rngs,
            )

        if getattr(config, "scan_layers", False):
            # Use nnx.vmap to create stacked layers, then nnx.scan for forward pass
            # All state is stacked on axis 0

            # Pre-split keys outside eval_shape to avoid RNG mutation in traced context
            keys = jax.random.split(rngs.params(), config.num_hidden_layers)
            @nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: 'layer'})
            def create_layer(key):
                return Qwen3DecoderLayer(config, dtype=dtype, rngs=nnx.Rngs(key), mesh=mesh)

            if mesh is None:
                raise ValueError("mesh must be provided when scan_layers=True to enable sharded initialization")

            # Use jax.jit with mesh context to create sharded layers directly
            # JAX will optimize away intermediate allocations and create sharded arrays
            @nnx.jit
            def create_layers():
                return create_layer(keys)

            with jax.set_mesh(mesh):
                self.layers = create_layers()

            # If reshape_for_scan is enabled, reshape layer state from [num_layers, ...] to [num_segments, segment_length, ...]
            # Otherwise, nnx.scan handles the reshaping internally.
            if getattr(config, 'reshape_for_scan', False) and config.segment_length is not None:
                num_segments = config.num_hidden_layers // config.segment_length
                segment_length = config.segment_length

                def reshape_to_segments(x):
                    if hasattr(x, 'shape') and hasattr(x, 'reshape') and x.ndim > 0:
                        new_shape = (num_segments, segment_length) + x.shape[1:]
                        return x.reshape(new_shape)
                    return x

                state = nnx.state(self.layers)
                reshaped_state = jax.tree.map(reshape_to_segments, state)
                nnx.update(self.layers, reshaped_state)
        else:
            self.layers = nnx.List(
                [Qwen3DecoderLayer(config, dtype=dtype, rngs=rngs, mesh=mesh) for _ in range(config.num_hidden_layers)]
            )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        output_hidden_states: bool | None = None,
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        
        
        
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        @nnx.remat(policy=jax.checkpoint_policies.nothing_saveable)
        def embed_tokens_forward(input_ids):
            return self.embed_tokens(input_ids)
        hidden_states = embed_tokens_forward(input_ids)
        all_hidden_states: list[jax.Array] = []
        
        # hidden_states = nn.with_logical_constraint(hidden_states, (("dp","layer"), None, None))

        if getattr(self.config, "scan_layers", False):
            # Use nnx.scan which handles state propagation and gradient flow correctly
            # Unlike jax.lax.scan + manual split/merge, nnx.scan preserves gradient paths
            num_layers = self.config.num_hidden_layers

            if False:
                pass # this commenting out is done for readability, we will uncomment it later, we aren't using it right now
                # stacked_keys = jnp.stack(kv_cache.keys, axis=0)
                # stacked_values = jnp.stack(kv_cache.values, axis=0)
                # cache_position = kv_cache.cache_position

                # # Scan Params and RngState on axis 0, broadcast everything else
                # layer_state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

                # @nnx.scan(
                #     in_axes=(layer_state_axes, nnx.Carry, None, None, 0, 0),
                #     out_axes=(nnx.Carry, (0, 0)),
                #     length=num_layers,  # Explicit length helps XLA recognize this as a loop
                #     transform_metadata={nnx.PARTITION_NAME: 'layer'},
                # )
                # @nnx.remat(policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
                # def apply_layer_with_cache(layer, h, attn_mask, pos, k_cache, v_cache):
                #     h, (k, v) = layer(
                #         h,
                #         attention_mask=attn_mask,
                #         positions=pos,
                #         kv_cache=(k_cache, v_cache, cache_position),
                #     )
                #     return h, (k, v)

                # hidden_states, (updated_keys, updated_values) = apply_layer_with_cache(
                #     self.layers, hidden_states, attention_mask, positions,
                #     stacked_keys, stacked_values
                # )
            else:
                # Scan Params and RngState on axis 0, broadcast everything else
                layer_state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

                def scan_fn(carry, layer):
                    h, attn_mask, pos = carry
                    input_dtype = h.dtype
                    h, _ = layer(
                        h,
                        attention_mask=attn_mask,
                        positions=pos,
                        kv_cache=None,
                    )
                    # Ensure output dtype matches input dtype for scan compatibility
                    h = h.astype(input_dtype)
                    new_carry = (h, attn_mask, pos)
                    return new_carry, None

                initial_carry = (hidden_states, attention_mask, positions)

                # Use segment_length for memory-efficient rematerialization
                # e.g., with 48 layers and segment_length=8, saves 6 boundary states instead of 48
                final_carry, _ = nnx.scan(
                    scan_fn,
                    length=num_layers,
                    in_axes=(nnx.Carry, 0),
                    # segment_length=1,  # 6 segments of 8 layers each
                    segment_broadcast_params=False,  # False=reshape approach, True=dynamic_slice
                )(initial_carry, self.layers)

                hidden_states, _, _ = final_carry

            # No KV cache needed during training (scan without cache)
            updated_keys = []
            updated_values = []

        else:
            # Original for-loop based forward pass
            updated_keys, updated_values = [], []

            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)

                hidden_states, (k, v) = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    positions=positions,
                    kv_cache=kv_cache and (kv_cache.keys[layer_idx], kv_cache.values[layer_idx], kv_cache.cache_position),
                )
                updated_keys.append(k)
                updated_values.append(v)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Increment cache_position if cache exists, or use sequence length for new cache
        new_cache_position = kv_cache.cache_position + 1 if kv_cache is not None else input_ids.shape[1]

        return ModelOutput(
            last_hidden_state=hidden_states,
            kv_cache=KVCache(keys=updated_keys, values=updated_values, cache_position=new_cache_position),
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class Qwen3ForCausalLM(nnx.Module, GeneratorMixin):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs, mesh=None) -> None:
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, rngs=rngs, mesh=mesh)
        if not self.config.tie_word_embeddings:
            embed_lora = getattr(config, "embed_lora", True)
            if embed_lora and config.lora_rank > 0:
                self.lm_head = LoRALinear(
                    config.hidden_size,
                    config.vocab_size,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=dtype,
                    kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, "tensor")),
                    lora_rank=config.lora_rank,
                    rngs=rngs,
                )
            else:
                self.lm_head = nnx.Linear(
                    config.hidden_size,
                    config.vocab_size,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=dtype,
                    kernel_init=nnx.with_metadata(nnx.initializers.lecun_normal(), sharding_names=(None, "tensor")),
                    rngs=rngs,
                )

    @staticmethod
    def is_lora_param(path: tuple, _value) -> bool:
        """Return True if a parameter path corresponds to LoRA weights."""
        return any(name in path for name in ("lora_a", "lora_b", "lora"))

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        kv_cache: KVCache | None = None,
    ) -> CausalLMOutput:
        if positions is None:
            positions = compute_positions(attention_mask)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            positions=positions,
            output_hidden_states=output_hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.embedding.value.T
        else:
            logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )


Qwen3MoeForCausalLM = Qwen3ForCausalLM

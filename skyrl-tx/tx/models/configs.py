"""Configuration classes for models with LoRA support."""

from transformers import PretrainedConfig


class Qwen3Config(PretrainedConfig):
    """Qwen3 configuration for tx.

    Wraps a HuggingFace PretrainedConfig with additional parameters
    for LoRA training and tensor parallelism.

    Args:
        config: A HuggingFace PretrainedConfig object (e.g., from Qwen3Config.from_pretrained())
        lora_rank: Rank for LoRA adapters (0 to disable LoRA)
        shard_attention_heads: Whether to shard attention across tensor parallel devices
        mlp_lora: Whether to enable LoRA for MLP layers
        attn_lora: Whether to enable LoRA for attention layers
        scan_layers: Whether to use scan over layers
    """

    # Type hints for LoRA attributes
    lora_rank: int
    shard_attention_heads: bool
    mlp_lora: bool
    attn_lora: bool
    scan_layers: bool

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        max_lora_rank: int = 0,
        shard_attention_heads: bool = True,
        mlp_lora: bool = True,
        attn_lora: bool = True,
        scan_layers: bool = False,
    ):
        # Copy all attributes from the base config
        super().__init__(**config.to_dict())

        # Add LoRA-specific parameters (max_lora_rank maps to lora_rank for NNX LoRA)
        self.lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.mlp_lora = mlp_lora
        self.attn_lora = attn_lora
        self.scan_layers = scan_layers

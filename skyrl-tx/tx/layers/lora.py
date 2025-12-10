"""LoRA layers using Flax NNX LoRA implementation."""

from flax import nnx
from flax.nnx.nn.lora import LoRA, LoRALinear, LoRAParam
import jax
from jax import numpy as jnp

from tx.models.types import ModelForCausalLM
from tx.tinker.types import LoraConfig


class LoRAEmbed(nnx.Embed):
    """An nnx.Embed layer with LoRA support."""

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        lora_rank: int = 0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype | None = None,
        embedding_init: nnx.Initializer,
        rngs: nnx.Rngs,
    ) -> None:
        param_dtype = param_dtype or dtype

        super().__init__(
            num_embeddings=num_embeddings,
            features=features,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=embedding_init,
            rngs=rngs,
        )

        if lora_rank > 0:
            # LoRA for embeddings: lora_a is indexed by token, lora_b projects to output
            self.lora = LoRA(
                in_features=num_embeddings,
                lora_rank=lora_rank,
                out_features=features,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.lora = None

    def __call__(self, x: jax.Array) -> jax.Array:
        base_out = super().__call__(x)
        if self.lora is not None:
            # For embeddings, we use one-hot encoding to apply LoRA
            one_hot = jax.nn.one_hot(x, self.num_embeddings, dtype=base_out.dtype)
            lora_out = self.lora(one_hot)
            return base_out + lora_out
        return base_out


# Re-export NNX LoRA classes for convenience
__all__ = ["LoRA", "LoRALinear", "LoRAParam", "LoRAEmbed"]

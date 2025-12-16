"""Abstract backend interface for TinkerEngine.

Backends handle all model state and computation. The engine handles file I/O and database operations.

  1. AbstractBackend (backend.py)

  Clean interface defining what backends must implement:
  - precompile_kernels, create_optimizer
  - process_forward_backward_batch, process_optim_step, process_sample_batch
  - extract_checkpoint_data, insert_checkpoint_data (pure state manipulation)
  - extract_sampler_weights, insert_sampler_weights

  2. MaxTextBackend (maxtext.py)

  - Implements all abstract methods
  - process_sample_batch -> raises NotImplementedError
  - insert_checkpoint_data -> raises NotImplementedError
  - insert_sampler_weights -> raises NotImplementedError
  - Added parse_maxtext_config() function

  3. NativeBackend (native.py)

  - Implements all abstract methods fully
  - extract_checkpoint_data / insert_checkpoint_data - pure state extraction/insertion
  - extract_sampler_weights / insert_sampler_weights - pure state manipulation

  4. TinkerEngine (engine.py)

  - Instantiates MaxTextBackend or NativeBackend based on config
  - Delegates computation to self.backend
  - Handles all file I/O (download_and_unpack, pack_and_upload)
  - Handles all database operations
"""

from abc import ABC, abstractmethod

import jax
from flax import nnx

from tx.tinker import types
from tx.tinker.config import EngineConfig


class AbstractBackend(ABC):
    """Abstract base class for TinkerEngine backends.

    Backends handle pure computation and model state manipulation.
    File I/O and database operations are handled by TinkerEngine.
    """

    config: EngineConfig
    mesh: jax.sharding.Mesh
    model: nnx.Module
    metrics: types.EngineMetrics
    graphdef: nnx.GraphDef
    lora_params: nnx.State
    non_lora_params: nnx.State

    @abstractmethod
    def __init__(self, config: EngineConfig, **kwargs):
        """Initialize the backend."""
        pass

    @abstractmethod
    def precompile_kernels(self, seq_lens: list[int]) -> None:
        """Precompile JIT kernels for specified sequence lengths."""
        pass

    @abstractmethod
    def create_optimizer(self, model_id: str) -> nnx.Optimizer:
        """Create an optimizer for a model."""
        pass

    @abstractmethod
    def process_forward_backward_batch(
        self,
        requests: dict[str, tuple[str, types.ForwardBackwardInput]],
        models: dict[str, types.ModelMetadata],
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward_backward requests in a batch."""
        pass

    @abstractmethod
    def process_optim_step(
        self,
        model_id: str,
        request_data: types.OptimStepInput,
        optimizer: nnx.Optimizer,
        adapter_index: int | None = None,
    ) -> types.OptimStepOutput:
        """Process an optimizer step request."""
        pass

    @abstractmethod
    def process_sample_batch(
        self,
        requests: dict[str, tuple[str, types.SampleInput]],
        models: dict[str, types.ModelMetadata],
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process multiple sample requests in a single batch."""
        pass

    @abstractmethod
    def extract_checkpoint_data(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
        optimizers: dict[str, nnx.Optimizer],
    ) -> dict:
        """Extract model state for checkpointing.

        Returns:
            Dictionary containing checkpoint data (weights, optimizer state, config).
        """
        pass

    @abstractmethod
    def insert_checkpoint_data(
        self,
        model_id: str,
        checkpoint_data: dict,
        models: dict[str, types.ModelMetadata],
        optimizers: dict[str, nnx.Optimizer],
    ) -> None:
        """Insert checkpoint data into model state.

        Args:
            model_id: The model identifier
            checkpoint_data: Dictionary from extract_checkpoint_data or loaded from disk
            models: Dict mapping model_id to ModelMetadata
            optimizers: Dict mapping model_id to Optimizer
        """
        pass

    @abstractmethod
    def extract_sampler_weights(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract weights for sampler checkpoint.

        Returns:
            Dictionary containing sampler weights data.
        """
        pass

    @abstractmethod
    def insert_sampler_weights(
        self,
        model_id: str,
        checkpoint_id: str,
        weights_data: dict,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert sampler weights into model state.

        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            weights_data: Dictionary from extract_sampler_weights or loaded from disk
            models: Dict mapping model_id to ModelMetadata
        """
        pass

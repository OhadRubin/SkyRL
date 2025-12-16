"""Background engine for processing training requests."""

import argparse
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel
from sqlmodel import create_engine, Session, select, update, func

from flax import nnx
from flax.training import checkpoints

from tx.tinker.db_models import FutureDB, RequestStatus, CheckpointDB, CheckpointStatus
from tx.tinker import types
from tx.tinker.config import EngineConfig, add_model
from tx.tinker.backends import AbstractBackend, MaxTextBackend, NativeBackend, parse_maxtext_config
from tx.utils.storage import download_and_unpack, pack_and_upload
from tx.utils.models import save_lora_checkpoint, convert_maxtext_lora_to_hf
from tx.utils.log import logger


class TinkerEngine:
    """Background engine for processing training requests.

    Handles file I/O, database operations, and request scheduling.
    Delegates computation to the appropriate backend (MaxTextBackend or NativeBackend).
    """

    def __init__(
        self,
        config: EngineConfig,
    ):
        """Initialize the engine with a database connection and backend."""
        self.config = config
        self.db_engine = create_engine(config.database_url, echo=False)

        # Store LoRA model metadata (model_id -> metadata)
        self.models: dict[str, types.ModelMetadata] = {}
        # Store optimizer instances per LoRA adapter (model_id -> optimizer)
        self.optimizers: dict[str, nnx.Optimizer] = {}

        # Parse MaxText config if provided (for context parallelism)
        self.maxtext_config = parse_maxtext_config(self.config.maxtext_config_str)

        # Instantiate appropriate backend
        if self.maxtext_config:
            logger.info(f"Using MaxText backend with config: {self.config.maxtext_config_str}")
            self.backend: AbstractBackend = MaxTextBackend(config, self.maxtext_config)
        else:
            logger.info("Using Native backend")
            self.backend: AbstractBackend = NativeBackend(config)

        # Precompile kernels if requested
        if self.config.precompile_seq_lens:
            seq_lens = [int(s.strip()) for s in self.config.precompile_seq_lens.split(",") if s.strip()]
            self.backend.precompile_kernels(seq_lens)

    @contextmanager
    def _checkpoint_status_context(self, model_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType):
        """Context manager to handle checkpoint DB status updates.

        Fetches the checkpoint entry, yields it, and updates its status to COMPLETED
        or FAILED based on whether an exception occurred.
        """
        with Session(self.db_engine) as session:
            checkpoint_db = session.get(CheckpointDB, (model_id, checkpoint_id, checkpoint_type))
            if checkpoint_db is None:
                raise ValueError(
                    f"Checkpoint entry not found for model '{model_id}', checkpoint '{checkpoint_id}', type '{checkpoint_type}'"
                )

            try:
                yield checkpoint_db
                checkpoint_db.status = CheckpointStatus.COMPLETED
            except Exception as e:
                logger.exception(f"Error saving checkpoint for model {model_id}, checkpoint {checkpoint_id}: {e}")
                checkpoint_db.status = CheckpointStatus.FAILED
                checkpoint_db.error_message = str(e)
                raise
            finally:
                checkpoint_db.completed_at = datetime.now(timezone.utc)
                session.add(checkpoint_db)
                session.commit()

    def find_batchable_forward_backward(self, session: Session) -> dict[str, tuple[str, types.ForwardBackwardInput]]:
        """Find all forward_backward ops that come before any destructive update for their model.

        Uses look-ahead scheduling: for each model, only returns forward_backward operations
        that have no optim_step or load_weights blocking them in the queue.

        Args:
            session: Database session

        Returns:
            Dict mapping request_id to (model_id, request_data) tuples
        """
        # Find the earliest pending optim_step or load_weights per model (these act as barriers)
        barriers_query = (
            select(FutureDB.model_id, func.min(FutureDB.request_id).label("barrier_id"))
            .where(
                (FutureDB.request_type == types.RequestType.OPTIM_STEP)
                | (FutureDB.request_type == types.RequestType.LOAD_WEIGHTS)
            )
            .where(FutureDB.status == RequestStatus.PENDING)
            .group_by(FutureDB.model_id)
        )
        barriers = dict(session.exec(barriers_query).all())

        # Get all pending forward_backward operations ordered by request_id
        fwd_bwd_query = (
            select(FutureDB)
            .where(FutureDB.request_type == types.RequestType.FORWARD_BACKWARD)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        fwd_bwd_ops = session.exec(fwd_bwd_query).all()

        # Filter: only include ops that come before their model's barrier
        batchable = [op for op in fwd_bwd_ops if op.model_id not in barriers or op.request_id < barriers[op.model_id]]

        return {
            f.request_id: (f.model_id, types.ForwardBackwardInput.model_validate(f.request_data)) for f in batchable
        }

    def find_batchable_sample(self, session: Session) -> dict[str, tuple[str, types.SampleInput]]:
        """Find all sample ops that can be safely batched together.

        Returns sample operations ensuring that each model_id has only one checkpoint_id
        to avoid loading different checkpoints for the same model in a single batch.

        Args:
            session: Database session

        Returns:
            Dict mapping request_id to (model_id, request_data) tuples
        """
        sample_query = (
            select(FutureDB)
            .where(FutureDB.request_type == types.RequestType.SAMPLE)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        sample_ops = session.exec(sample_query).all()

        batchable = []
        model_checkpoints = {}  # Map from model_id to checkpoint_id of first request to that model
        for op in sample_ops:
            checkpoint_id = op.request_data["checkpoint_id"]
            # Base model requests (empty checkpoint_id) are always compatible, otherwise only
            # take only requests with one checkpoint_id for a given model_id
            if checkpoint_id == "" or model_checkpoints.setdefault(op.model_id, checkpoint_id) == checkpoint_id:
                batchable.append(op)

        return {f.request_id: (f.model_id, types.SampleInput.model_validate(f.request_data)) for f in batchable}

    def find_single_requests(self, session: Session) -> dict[str, tuple[str, types.RequestType, dict]]:
        """Find all requests that need to be processed individually (not batchable).

        Args:
            session: Database session

        Returns:
            Dict mapping request_id to (model_id, request_type, request_data) tuples
        """
        statement = (
            select(FutureDB)
            .where(FutureDB.status == RequestStatus.PENDING)
            .where(FutureDB.request_type != types.RequestType.FORWARD_BACKWARD)
            .where(FutureDB.request_type != types.RequestType.SAMPLE)
            .where(FutureDB.request_type != types.RequestType.EXTERNAL)
            .order_by(FutureDB.request_id)
        )
        other_futures = session.exec(statement).all()

        return {f.request_id: (f.model_id, f.request_type, f.request_data) for f in other_futures}

    def process_create_model(self, model_id: str, request_data: types.CreateModelInput) -> types.CreateModelOutput:
        """Create and initialize a model."""
        # Assign adapter index for this model_id
        adapter_index = max((m.adapter_index for m in self.models.values()), default=0) + 1

        # Extract LoRA configuration
        lora_config = request_data.lora_config

        # Validate rank doesn't exceed max
        if not (0 < lora_config.rank <= self.config.max_lora_rank):
            raise ValueError(f"LoRA rank {lora_config.rank} must be between 1 and {self.config.max_lora_rank}")

        self.models[model_id] = types.ModelMetadata(
            adapter_index=adapter_index,
            lora_config=lora_config,
        )

        # Create optimizer via backend
        self.optimizers[model_id] = self.backend.create_optimizer(model_id)

        logger.info(f"Created LoRA model {model_id} with adapter index {adapter_index}, config {lora_config}")

        return types.CreateModelOutput(
            model_id=model_id,
            base_model=self.config.base_model,
            lora_config=request_data.lora_config,
        )

    def process_forward_backward_batch(
        self, requests: dict[str, tuple[str, types.ForwardBackwardInput]]
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process multiple forward_backward requests in a single batch.

        Delegates to the backend for computation.
        """
        return self.backend.process_forward_backward_batch(requests, self.models)

    def process_sample_batch(
        self, requests: dict[str, tuple[str, types.SampleInput]]
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process multiple sample requests in a single batch.

        Handles loading sampler weights from disk (file I/O), then delegates
        computation to the backend.
        """
        # Load sampler weights from disk if needed (file I/O is engine responsibility)
        self._load_sampler_weights_for_requests(requests)

        # Delegate computation to backend
        return self.backend.process_sample_batch(requests, self.models)

    def process_optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Process an optim_step request and apply accumulated gradients.

        Delegates to the backend for computation.
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not loaded, skipping optimizer step")
            return types.OptimStepOutput()

        adapter_index = self.models[model_id].adapter_index
        return self.backend.process_optim_step(
            model_id, request_data, self.optimizers[model_id], adapter_index
        )

    def process_load_weights(self, model_id: str, request_data: types.LoadWeightsInput) -> types.LoadWeightsOutput:
        """Loads a clean, trimmed training checkpoint.

        Handles file I/O (download), delegates state insertion to backend.
        """
        if model_id not in self.models:
            raise ValueError("Model not loaded. Create the model before loading a checkpoint.")

        checkpoint_dir = (
            self.config.checkpoints_base / request_data.source_model_id / f"{request_data.checkpoint_id}.tar.gz"
        )

        # Download and extract checkpoint (file I/O)
        with download_and_unpack(checkpoint_dir) as temp_dir:
            # Get empty checkpoint structure from backend for restoration target
            checkpoint_data = self.backend.extract_checkpoint_data(model_id, self.models, self.optimizers)
            checkpoint = checkpoints.restore_checkpoint(
                ckpt_dir=temp_dir, target=checkpoint_data, prefix="checkpoint_"
            )

        if checkpoint is None:
            raise FileNotFoundError(f"Training checkpoint not found in {checkpoint_dir}")

        # Delegate state insertion to backend
        self.backend.insert_checkpoint_data(model_id, checkpoint, self.models, self.optimizers)

        logger.info(f"Loaded training checkpoint for model {model_id} from {checkpoint_dir}")
        return types.LoadWeightsOutput(type="load_weights")

    def process_save_weights(self, model_id: str, request_data: types.SaveWeightsInput) -> types.SaveWeightsOutput:
        """Saves a clean training checkpoint.

        Handles file I/O (upload), delegates state extraction to backend.
        For MaxText: saves in HuggingFace PEFT format.
        For Native: saves using Flax checkpoints format.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        checkpoint_id = request_data.path

        if self.maxtext_config:
            # MaxText path: save in HuggingFace PEFT format
            output_path = self.config.checkpoints_base / model_id / checkpoint_id

            with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.TRAINING):
                checkpoint_data = self.backend.extract_checkpoint_data(model_id, self.models, self.optimizers)
                convert_maxtext_lora_to_hf(
                    lora_state=checkpoint_data["lora_params"],
                    output_path=output_path,
                    base_model_name=self.config.base_model,
                    lora_rank=checkpoint_data["lora_rank"],
                    lora_alpha=checkpoint_data["lora_alpha"],
                )
                logger.info(f"Saved MaxText LoRA checkpoint in HF format for model {model_id} to {output_path}")

            return types.SaveWeightsOutput(
                path=f"tinker://{model_id}/weights/{checkpoint_id}",
                type="save_weights",
            )
        else:
            # Native path: save using Flax checkpoints
            output_path = self.config.checkpoints_base / model_id / f"{checkpoint_id}.tar.gz"

            with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.TRAINING):
                with pack_and_upload(output_path) as temp_dir:
                    checkpoint_data = self.backend.extract_checkpoint_data(model_id, self.models, self.optimizers)
                    checkpoints.save_checkpoint(
                        target=checkpoint_data,
                        ckpt_dir=temp_dir,
                        step=0,
                        prefix="checkpoint_",
                        overwrite=True,
                    )

                logger.info(f"Saved trimmed training checkpoint for model {model_id} to {output_path}")

            return types.SaveWeightsOutput(
                path=f"tinker://{model_id}/weights/{checkpoint_id}",
                type="save_weights",
            )

    def process_save_weights_for_sampler(
        self, model_id: str, request_data: types.SaveWeightsForSamplerInput
    ) -> types.SaveWeightsForSamplerOutput:
        """Save model weights for sampler checkpoint.

        Handles file I/O (upload), delegates state extraction to backend.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        lora_model = self.models[model_id]

        # Make sure the user cannot store checkpoints in places like ../../<important file>
        checkpoint_id = Path(request_data.path).name

        if self.maxtext_config:
            # MaxText path: save in HuggingFace PEFT format
            output_path = self.config.checkpoints_base / model_id / "sampler_weights" / checkpoint_id

            with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.SAMPLER):
                checkpoint_data = self.backend.extract_sampler_weights(model_id, self.models)
                convert_maxtext_lora_to_hf(
                    lora_state=checkpoint_data["lora_params"],
                    output_path=output_path,
                    base_model_name=self.config.base_model,
                    lora_rank=checkpoint_data["lora_rank"],
                    lora_alpha=checkpoint_data["lora_alpha"],
                )
                logger.info(
                    f"Saved MaxText LoRA sampler checkpoint in HF format for model {model_id} to {output_path}"
                )
        else:
            # Native path: save using save_lora_checkpoint
            output_path = self.config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"

            with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.SAMPLER):
                # Get weights data from backend
                weights_data = self.backend.extract_sampler_weights(model_id, self.models)
                # Save the LoRA adapter weights and LoRA config as tar.gz
                save_lora_checkpoint(
                    weights_data["model"],
                    weights_data["base_model"],
                    weights_data["lora_config"],
                    weights_data["adapter_index"],
                    output_path
                )

                logger.info(
                    f"Saved LoRA adapter weights for model {model_id} (adapter {lora_model.adapter_index}) to {output_path}"
                )

        return types.SaveWeightsForSamplerOutput(
            path=f"tinker://{model_id}/{checkpoint_id}",
            type="save_weights_for_sampler",
        )

    def _load_sampler_weights_for_requests(self, requests: dict[str, tuple[str, types.SampleInput]]) -> None:
        """Load sampler weights from disk for requests that need them.

        This is the file I/O portion - delegates state insertion to the backend.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples for the batch
        """
        for _, (model_id, request_data) in requests.items():
            base_model = request_data.base_model
            checkpoint_id = request_data.checkpoint_id

            if base_model is None:
                # This code path is for sampling from a LoRA adapter
                assert checkpoint_id != "", "checkpoint_id must be not empty"

                if self.models[model_id].loaded_checkpoint_id != checkpoint_id:
                    # Load model from disk and insert into backend state
                    checkpoint_path = (
                        self.config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
                    )
                    logger.info(f"Loading LoRA sampler checkpoint from {checkpoint_path}")
                    self.backend.insert_sampler_weights(
                        model_id, checkpoint_id, checkpoint_path, self.models
                    )
            else:
                # This code path is for sampling from the base model
                if base_model != self.config.base_model:
                    raise ValueError(
                        f"Requested base_model '{base_model}' does not match engine's base_model '{self.config.base_model}'"
                    )

    def _complete_futures(self, results: dict[str, BaseModel]):
        """Helper method to complete multiple futures in the database.

        Args:
            results: Dict mapping request_id to result (Pydantic BaseModel)
        """
        completed_at = datetime.now(timezone.utc)
        params = [
            {
                "request_id": int(request_id),
                "result_data": result.model_dump(),
                "status": RequestStatus.FAILED if isinstance(result, types.ErrorResponse) else RequestStatus.COMPLETED,
                "completed_at": completed_at,
            }
            for request_id, result in results.items()
        ]

        with Session(self.db_engine) as session:
            session.execute(update(FutureDB), params)
            session.commit()

    def process_single_request(self, request_type: types.RequestType, model_id: str, request_data: dict) -> BaseModel:
        match request_type:
            case types.RequestType.CREATE_MODEL:
                return self.process_create_model(model_id, types.CreateModelInput.model_validate(request_data))
            case types.RequestType.OPTIM_STEP:
                return self.process_optim_step(model_id, types.OptimStepInput.model_validate(request_data))
            case types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER:
                return self.process_save_weights_for_sampler(
                    model_id, types.SaveWeightsForSamplerInput.model_validate(request_data)
                )
            case types.RequestType.SAVE_WEIGHTS:
                return self.process_save_weights(model_id, types.SaveWeightsInput.model_validate(request_data))
            case types.RequestType.LOAD_WEIGHTS:
                pass
                # return self.process_load_weights(model_id, types.LoadWeightsInput.model_validate(request_data))
            case _:
                raise ValueError(f"Unknown request type: {request_type}")

    def process_batch_requests(self, requests: dict[str, tuple[str, BaseModel]], batch_processor):
        """Generic function to process a batch of requests.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples
            batch_processor: Function to call to process the batch (e.g., process_forward_backward_batch)
        """
        if not requests:
            return
        try:
            results = batch_processor(requests)
            self._complete_futures(results)
        except Exception as e:
            logger.exception(f"Error processing batch: {e}")
            self._complete_futures(
                {request_id: types.ErrorResponse(error=str(e), status="failed") for request_id in requests}
            )

    def process_pending_requests(self):
        """Main loop to process pending requests."""
        while True:
            # Query for pending requests and extract data within session context
            with Session(self.db_engine) as session:
                # Use look-ahead scheduling to find batchable forward_backward operations
                forward_backward_requests = self.find_batchable_forward_backward(session)
                # Find pending sample requests that can be batched
                sample_requests = self.find_batchable_sample(session)
                # Get other pending requests (non forward_backward and non sampling)
                other_requests = self.find_single_requests(session)

            # Process batches outside of session context
            self.process_batch_requests(forward_backward_requests, self.process_forward_backward_batch)
            self.process_batch_requests(sample_requests, self.process_sample_batch)

            # Process other request types individually (in the future we can also batch independent optim_steps)
            other_results = {}
            for request_id, (model_id, request_type, request_data) in other_requests.items():
                try:
                    result = self.process_single_request(request_type, model_id, request_data)
                except Exception as e:
                    logger.exception(f"Error processing request {request_id}: {e}")
                    result = types.ErrorResponse(error=str(e), status="failed")
                other_results[request_id] = result

            self._complete_futures(other_results)

            # Poll every 100ms
            time.sleep(0.1)

    def run(self):
        """Entry point to start the engine."""
        logger.info("Starting background engine...")
        self.process_pending_requests()


def main():
    """Entry point for the background engine."""
    # Create argument parser and add Pydantic model fields
    parser = argparse.ArgumentParser(description="SkyRL tx tinker engine for processing requests")
    add_model(parser, EngineConfig)

    # Parse command-line arguments
    args = parser.parse_args()

    # Create EngineConfig from parsed arguments
    config = EngineConfig.model_validate(vars(args))

    # Initialize and run the engine
    TinkerEngine(config).run()


if __name__ == "__main__":
    main()

import asyncio
import shutil
import sys

import httpx
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential
from datetime import datetime, timezone
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import OperationalError


from observability import log, bootstrap, set_run_id, Events

from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.db_models import FutureDB, RequestStatus
from tx.utils.storage import download_and_unpack




class AllServersFailed(Exception):
    """Raised when all servers fail in a single cycle."""

    def __init__(self, message: str, is_timeout: bool):
        super().__init__(message)
        self.is_timeout = is_timeout


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., vLLM)."""

    MAX_PREFIX_LEVELS = 4  # Try up to 4 prefix granularities for hierarchical matching
    MIN_PREFIX_CHUNKS = 3  # Skip prefixes shorter than this (avoids matching on just system prompt)
    MAX_CONCURRENT_REQUESTS = 1000  # Limit concurrent background tasks to prevent pool exhaustion
    MAX_PENDING_TASKS = 2000  # Maximum queued tasks before rejecting new requests

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.router_url = engine_config.external_inference_urls[0]
        self.api_key = engine_config.external_inference_api_key
        self.checkpoints_base = engine_config.checkpoints_base
        self.lora_base_dir = engine_config.external_inference_lora_base
        self.db_engine = db_engine
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self._pending_count = 0
        self._pending_lock = asyncio.Lock()

    @property
    def cache_stats(self) -> str:
        return "handled by router"

    async def check_capacity(self) -> bool:
        """Check if we can accept more requests. Returns True if under capacity."""
        async with self._pending_lock:
            return self._pending_count < self.MAX_PENDING_TASKS

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
    ):
        """Background task to call external engine and store result in database."""
        async with self._pending_lock:
            self._pending_count += 1

        try:
            async with self._semaphore:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "X-Job-ID": model_id,
                    }
                    if sample_req.session_id:
                        headers["X-Tx-Param-Session-ID"] = sample_req.session_id

                    async with httpx.AsyncClient(
                        base_url=self.router_url,
                        headers=headers,
                        timeout=httpx.Timeout(300.0, connect=10.0),  # 5 minutes for inference, 10s for connect
                    ) as http_client:
                        result = await self._forward_to_engine(sample_req, model_id, checkpoint_id, http_client)
                    result_data = result.model_dump()
                    status = RequestStatus.COMPLETED
                except Exception as e:
                    log.error("unexpected error in external inference", component="external_inference", error=str(e))
                    result_data = {"error": str(e), "status": "failed"}
                    status = RequestStatus.FAILED

                await self._store_result_with_retry(request_id, result_data, status)
        finally:
            async with self._pending_lock:
                self._pending_count -= 1

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=5),
        retry=retry_if_exception_type(OperationalError),
        reraise=True,
    )
    async def _store_result_with_retry(self, request_id: int, result_data: dict, status: RequestStatus):
        """Store result in DB with retry for SQLite lock contention."""
        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_to_engine(
        self, request, model_id: str, checkpoint_id: str, http_client: httpx.AsyncClient
    ) -> types.SampleOutput:
        """Forward request to vLLM with dynamic LoRA loading.

        Extracts the checkpoint to the configured external_inference_lora_base and references it by a model name
        that vLLM can dynamically load via the lora_filesystem_resolver plugin.
        """
        prompt_tokens = [token for chunk in request.prompt.chunks for token in chunk.tokens]
        checkpoint_path = self.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
        model_name = f"{model_id}_{checkpoint_id}"
        target_dir = self.lora_base_dir / model_name
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        # Extract the checkpoint if it doesn't already exist
        if not target_dir.exists():
            try:
                with download_and_unpack(checkpoint_path) as extracted_path:
                    # shutil.move allows moving between filesystems.
                    shutil.move(str(extracted_path), str(target_dir))
            except FileExistsError:
                # Race condition: another process created target_dir
                pass

        payload = {
            "model": model_name,
            "prompt": prompt_tokens,
            "max_tokens": request.sampling_params.max_tokens,
            "temperature": request.sampling_params.temperature,
            "logprobs": True,
            "stream": False,
            "return_token_ids": True,
        }

        # Request prompt logprobs if needed (vLLM extra parameter)
        if request.prompt_logprobs:
            payload["prompt_logprobs"] = 1  # Get top-1 logprob per prompt token

        response = await http_client.post("/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        sequences = []
        prompt_logprobs_out: list[float | None] = []

        for choice in result["choices"]:
            lp = choice["logprobs"]
            sequences.append(
                types.GeneratedSequence(
                    tokens=choice["token_ids"],
                    logprobs=lp["token_logprobs"],
                    stop_reason=choice["finish_reason"],
                )
            )

            # Extract prompt logprobs if present (vLLM returns these at choice level, not inside logprobs)
            if request.prompt_logprobs and "prompt_logprobs" in choice and choice["prompt_logprobs"]:
                # vLLM returns list of dicts with token -> logprob mappings
                # Extract the logprob for each actual prompt token
                for i, token_probs in enumerate(choice["prompt_logprobs"]):
                    if token_probs is None:
                        prompt_logprobs_out.append(None)
                    elif isinstance(token_probs, dict):
                        # Get the logprob for the actual token at this position
                        prompt_token = prompt_tokens[i] if i < len(prompt_tokens) else None
                        if prompt_token is not None and str(prompt_token) in token_probs:
                            prompt_logprobs_out.append(token_probs[str(prompt_token)]["logprob"])
                        elif token_probs:
                            # Fallback: take the first (highest prob) token's logprob
                            first_key = next(iter(token_probs))
                            prompt_logprobs_out.append(token_probs[first_key]["logprob"])
                        else:
                            prompt_logprobs_out.append(None)
                    else:
                        # Direct float value
                        prompt_logprobs_out.append(float(token_probs))

        return types.SampleOutput(sequences=sequences, prompt_logprobs=prompt_logprobs_out)

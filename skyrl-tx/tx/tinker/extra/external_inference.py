import asyncio
import shutil

import httpx
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from datetime import datetime, timezone
from sqlmodel.ext.asyncio.session import AsyncSession

from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.db_models import FutureDB, RequestStatus
from tx.utils.log import logger
from tx.utils.storage import download_and_unpack


TIMEOUT_ERROR_PREFIX = "TINKER_TIMEOUT: "
UNHEALTHY_DURATION_SECONDS = 120  # 2 minutes


class AllServersFailed(Exception):
    """Raised when all servers fail in a single cycle."""

    def __init__(self, message: str, is_timeout: bool):
        super().__init__(message)
        self.is_timeout = is_timeout


class ExternalInferenceClient:
    """Client for calling external inference engines (e.g., vLLM)."""

    MAX_PREFIX_LEVELS = 4  # Try up to 4 prefix granularities for hierarchical matching
    MIN_PREFIX_CHUNKS = 3  # Skip prefixes shorter than this (avoids matching on just system prompt)

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.base_urls = [f"{url}/v1" for url in engine_config.external_inference_urls]
        self._url_index = 0
        self.api_key = engine_config.external_inference_api_key
        self.checkpoints_base = engine_config.checkpoints_base
        self.lora_base_dir = engine_config.external_inference_lora_base
        self.db_engine = db_engine
        self._prefix_to_server: dict[int, str] = {}  # Stateful: prefix_hash → server URL
        self._cache_hits = 0
        self._cache_misses = 0
        self._unhealthy_urls: dict[str, datetime] = {}  # url → timestamp when marked unhealthy

    @property
    def cache_stats(self) -> str:
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return "no requests yet"
        miss_rate = self._cache_misses / total
        return f"miss_rate={miss_rate:.2%} ({self._cache_misses}/{total})"

    @property
    def unhealthy_urls(self) -> list[str]:
        """Return list of currently unhealthy URLs."""
        return [url for url in self._unhealthy_urls if not self._is_url_healthy(url)]

    def _is_url_healthy(self, url: str) -> bool:
        """Check if URL is healthy (not marked unhealthy or unhealthy period expired)."""
        if url not in self._unhealthy_urls:
            return True
        marked_at = self._unhealthy_urls[url]
        elapsed = (datetime.now(timezone.utc) - marked_at).total_seconds()
        if elapsed >= UNHEALTHY_DURATION_SECONDS:
            del self._unhealthy_urls[url]
            logger.info(f"URL {url} recovered after {elapsed:.1f}s unhealthy period")
            return True
        return False

    def _mark_url_unhealthy(self, url: str):
        """Mark URL as unhealthy for UNHEALTHY_DURATION_SECONDS."""
        self._unhealthy_urls[url] = datetime.now(timezone.utc)
        logger.warning(f"Marked {url} as unhealthy for {UNHEALTHY_DURATION_SECONDS}s")

    def _seconds_until_any_healthy(self) -> float | None:
        """Return seconds until the earliest unhealthy URL recovers, or None if any URL is already healthy."""
        if not self._unhealthy_urls:
            return None
        now = datetime.now(timezone.utc)
        min_wait = float("inf")
        for url, marked_at in self._unhealthy_urls.items():
            elapsed = (now - marked_at).total_seconds()
            remaining = UNHEALTHY_DURATION_SECONDS - elapsed
            if remaining <= 0:
                return None
            min_wait = min(min_wait, remaining)
        if min_wait == float("inf"):
            return None
        return min_wait

    def _get_next_url(self) -> str | None:
        """Round-robin server selection, skipping unhealthy URLs."""
        for _ in range(len(self.base_urls)):
            url = self.base_urls[self._url_index % len(self.base_urls)]
            self._url_index += 1
            if self._is_url_healthy(url):
                return url
        return None

    def _compute_prefix_hashes(self, sample_req) -> list[int]:
        """Compute hashes for prefix chunks at multiple granularities.

        Returns hashes for chunks[:-1], chunks[:-2], etc. (longest to shortest) for hierarchical matching.
        Skips prefixes shorter than MIN_PREFIX_CHUNKS to avoid matching on just system prompt.
        """
        chunks = sample_req.prompt.chunks
        hashes = []

        for skip in range(1, self.MAX_PREFIX_LEVELS + 1):
            if len(chunks) <= skip:
                break
            prefix_chunks = chunks[:-skip]
            if len(prefix_chunks) < self.MIN_PREFIX_CHUNKS:
                break
            prefix_tokens = tuple(t for chunk in prefix_chunks for t in chunk.tokens)
            if prefix_tokens:
                hashes.append(hash(prefix_tokens))

        # logger.info(f"Prefix hashes: num_chunks={len(chunks)}, num_hashes={len(hashes)}, hashes={hashes}")
        return hashes

    def _get_url_for_hashes(self, prefix_hashes: list[int]) -> str | None:
        """Hierarchical hash-based server selection for KV cache affinity.

        Tries each prefix hash (longest to shortest). If any matches and is healthy, route there (hit).
        If none match or cached URL is unhealthy, round-robin assign and store all hashes for future matching.
        Returns None if all URLs are unhealthy.
        """
        for i, h in enumerate(prefix_hashes):
            if h in self._prefix_to_server:
                cached_url = self._prefix_to_server[h]
                if self._is_url_healthy(cached_url):
                    self._cache_hits += 1
                    return cached_url

        self._cache_misses += 1
        url = self._get_next_url()

        if url is not None:
            for h in prefix_hashes:
                self._prefix_to_server[h] = url

        return url

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
    ):
        """Background task to call external engine and store result in database.

        Uses hash-based routing for KV cache affinity: requests with the same prefix
        are routed to the same server. Falls back to round-robin on connection failures.
        """
        prefix_hashes = self._compute_prefix_hashes(sample_req)

        # @retry(
        #     stop=stop_after_attempt(3),
        #     retry=retry_if_exception_type(AllServersFailed),
        #     reraise=True,
        # )
        async def _try_all_servers() -> types.SampleOutput:
            # May block waiting for unhealthy servers to recover; caller handles timeout.
            last_error: Exception | None = None
            is_timeout_error = False

            while True:
                tried_urls: set[str] = set()

                for attempt in range(len(self.base_urls)):
                    if attempt == 0 and prefix_hashes:
                        base_url = self._get_url_for_hashes(prefix_hashes)
                    else:
                        base_url = self._get_next_url()
                        while base_url is not None and base_url in tried_urls and len(tried_urls) < len(self.base_urls):
                            base_url = self._get_next_url()

                    if base_url is None:
                        break
                    if base_url in tried_urls:
                        continue
                    tried_urls.add(base_url)

                    try:
                        async with httpx.AsyncClient(
                            base_url=base_url,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            timeout=httpx.Timeout(300.0, connect=10.0),
                        ) as http_client:
                            result = await self._forward_to_engine(sample_req, model_id, checkpoint_id, http_client)
                        for h in prefix_hashes:
                            self._prefix_to_server[h] = base_url
                        return result
                    except httpx.HTTPStatusError as e:
                        logger.warning(f"Server {base_url} failed with HTTP {e.response.status_code}, trying next: {e}")
                        last_error = e
                        if e.response.status_code >= 500:
                            self._mark_url_unhealthy(base_url)
                        continue
                    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as e:
                        logger.warning(f"Server {base_url} failed, trying next: {e}")
                        last_error = e
                        is_timeout_error = isinstance(e, httpx.TimeoutException)
                        self._mark_url_unhealthy(base_url)
                        continue

                wait_seconds = self._seconds_until_any_healthy()
                if wait_seconds is None:
                    raise AllServersFailed(
                        f"All {len(self.base_urls)} servers failed, last error: {last_error}",
                        is_timeout=is_timeout_error,
                    )
                logger.info(f"All servers unhealthy, waiting {wait_seconds:.1f}s for recovery")
                await asyncio.sleep(wait_seconds + 0.1)

        try:
            result = await _try_all_servers()
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except AllServersFailed as e:
            logger.exception(f"All external engines failed after retries")
            error_msg = f"{TIMEOUT_ERROR_PREFIX}{e}" if e.is_timeout else str(e)
            result_data = {"error": error_msg, "status": "failed"}
            status = RequestStatus.FAILED
        except Exception as e:
            logger.exception(f"Unexpected error in external inference")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

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

        response = await http_client.post("/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        sequences = []
        for choice in result["choices"]:
            lp = choice["logprobs"]
            sequences.append(
                types.GeneratedSequence(
                    tokens=choice["token_ids"],
                    logprobs=lp["token_logprobs"],
                    stop_reason=choice["finish_reason"],
                )
            )

        return types.SampleOutput(sequences=sequences, prompt_logprobs=[])

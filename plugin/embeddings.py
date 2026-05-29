"""OpenAI-compatible embedding client for Qdrant memory plugin.

Features:
- Exponential backoff retry (3 attempts)
- Embedding dimension validation against VECTOR_DIM
- Circuit breaker (v0.7.0): tracks consecutive failures, pauses when threshold reached
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urljoin

from .config import VECTOR_DIM, BREAKER_THRESHOLD, BREAKER_COOLDOWN_SECS

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # seconds — exponential backoff


class EmbeddingCircuitBreaker:
    """Module-level circuit breaker for embedding API calls.

    Tracks consecutive failures. When threshold is reached, all calls
    fail fast until cooldown expires. Thread-safe via GIL (simple counter).
    """

    def __init__(
        self,
        threshold: int = BREAKER_THRESHOLD,
        cooldown_secs: float = BREAKER_COOLDOWN_SECS,
    ):
        self.threshold = threshold
        self.cooldown_secs = cooldown_secs
        self._failure_count = 0
        self._cooldown_until = 0.0

    @property
    def is_open(self) -> bool:
        """True if breaker is tripped (calls should fail fast)."""
        if self._failure_count < self.threshold:
            return False
        if time.monotonic() >= self._cooldown_until:
            # Cooldown expired — reset and allow calls
            self._failure_count = 0
            return False
        return True

    def record_success(self) -> None:
        """Reset failure count on successful call."""
        self._failure_count = 0

    def record_failure(self) -> None:
        """Increment failure count, trip breaker if threshold reached."""
        self._failure_count += 1
        if self._failure_count >= self.threshold:
            self._cooldown_until = time.monotonic() + self.cooldown_secs
            logger.warning(
                "Embedding circuit breaker tripped after %d consecutive failures. "
                "Pausing for %ds.",
                self._failure_count, self.cooldown_secs,
            )


# Module-level breaker instance
_breaker = EmbeddingCircuitBreaker()


def embed(texts: list[str], config: dict) -> list[list[float]]:
    """Get embeddings from an OpenAI-compatible API with retry and validation.

    Wrapped by module-level circuit breaker — if the embedding API is down,
    consecutive failures will trip the breaker and fail fast until cooldown.
    """
    import requests

    # Circuit breaker check
    if _breaker.is_open:
        raise RuntimeError(
            "Embedding API circuit breaker is open — too many consecutive failures. "
            "Will retry after cooldown."
        )

    base = config["embedding_base_url"].rstrip("/")
    url = urljoin(base + "/", "embeddings")

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {config['embedding_api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config["embedding_model"],
                    "input": texts,
                    "dimensions": VECTOR_DIM,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            vectors = [item["embedding"] for item in items]

            # Validate dimensions
            for i, vec in enumerate(vectors):
                if len(vec) != VECTOR_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(vec)}, expected {VECTOR_DIM}. "
                        f"Check EMBEDDING_MODEL and VECTOR_DIM config."
                    )

            _breaker.record_success()
            return vectors

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "Embedding attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Embedding failed after %d attempts: %s", MAX_RETRIES, e
                )

    _breaker.record_failure()
    raise last_error

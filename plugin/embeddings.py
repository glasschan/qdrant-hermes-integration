"""OpenAI-compatible embedding client for Qdrant memory plugin.

Features:
- Exponential backoff retry (3 attempts)
- Embedding dimension validation against VECTOR_DIM
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urljoin

from .config import VECTOR_DIM

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # seconds — exponential backoff


def embed(texts: list[str], config: dict) -> list[list[float]]:
    """Get embeddings from an OpenAI-compatible API with retry and validation."""
    import requests

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

    raise last_error

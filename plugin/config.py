"""Config loading for Qdrant memory plugin.

Loads config from env vars + $HERMES_HOME/qdrant-memory.json.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

VECTOR_DIM = 2048

# Circuit breaker
BREAKER_THRESHOLD = 5
BREAKER_COOLDOWN_SECS = 120

# Memory hygiene
DEDUP_THRESHOLD = 0.85          # cosine similarity for duplicate detection
DEDUP_ENABLED = True             # pre-save dedup on/off
AUTO_SYNC_CONVERSATIONS = False  # auto-save user messages to memory
SEARCH_RECENCY_WEIGHT = 0.0      # 0=pure relevance, 1=50/50 relevance+freshness


def load_config() -> dict:
    from hermes_constants import get_hermes_home

    config = {
        "qdrant_url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.environ.get("QDRANT_API_KEY", ""),
        "embedding_base_url": os.environ.get("EMBEDDING_BASE_URL", ""),
        "embedding_api_key": os.environ.get("EMBEDDING_API_KEY", ""),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "doubao-embedding-vision"),
        "collection_name": os.environ.get("QDRANT_COLLECTION", ""),
        "user_id": os.environ.get("QDRANT_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("QDRANT_AGENT_ID", "hermes"),
        # Memory hygiene settings
        "dedup_threshold": float(os.environ.get("QDRANT_DEDUP_THRESHOLD", str(DEDUP_THRESHOLD))),
        "dedup_enabled": os.environ.get("QDRANT_DEDUP_ENABLED", str(DEDUP_ENABLED)).lower() == "true",
        "auto_sync_conversations": os.environ.get("QDRANT_AUTO_SYNC", str(AUTO_SYNC_CONVERSATIONS)).lower() == "true",
        "search_recency_weight": float(os.environ.get("QDRANT_RECENCY_WEIGHT", str(SEARCH_RECENCY_WEIGHT))),
    }

    config_path = get_hermes_home() / "qdrant-memory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            logger.warning("Failed to load config from %s", config_path, exc_info=False)

    return config

"""Config loading for Qdrant memory plugin.

Loads config from env vars + $HERMES_HOME/qdrant-memory.json.
v0.7.0+: Config validation, new constants for priority filter, prefetch
         min turns, auto-stale/prune.
v0.9.0+: Full config validation with range checks.
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

# Auto-context injection (prefetch)
PREFETCH_TOP_K = 8              # how many memories to surface per turn
PREFETCH_SCORE_THRESHOLD = 0.4  # skip results below this cosine score
PREFETCH_MIN_TURNS = 3          # skip prefetch on conversations with fewer user turns
PREFETCH_CATEGORY_BOOST = {     # multiply score for high-priority categories
    "correction": 1.3,
    "instruction": 1.2,
    "preference": 1.15,
}

# Search priority filter
SEARCH_MIN_PRIORITY = 1          # 1 = show all, 5 = only highest quality

# Consolidation + evolution
CONSOLIDATION_CORRECTION_TOPIC_THRESHOLD = 3  # N corrections on same topic → suggest skill
CONSOLIDATION_ORPHAN_AGE_DAYS = 60            # days without being searched = orphan

# Session-end auto-extraction
SESSION_END_AUTO_EXTRACT = True

# Memory lifecycle (v0.9.0)
AUTO_STALE_ENABLED = False       # auto-bump stale low-priority memories to priority 5
AUTO_PRUNE_ENABLED = False       # auto-delete priority-5 memories older than 180 days
AUTO_STALE_DAYS = 90             # days before stale check triggers
AUTO_PRUNE_DAYS = 180            # days before prune check triggers

# Incremental consolidation
CONSOLIDATION_METADATA_ID = "00000000-0000-0000-0000-000000000001"  # fixed UUID for consolidation metadata


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
        # Auto-context injection
        "prefetch_top_k": int(os.environ.get("QDRANT_PREFETCH_TOP_K", str(PREFETCH_TOP_K))),
        "prefetch_score_threshold": float(os.environ.get("QDRANT_PREFETCH_SCORE_THRESHOLD", str(PREFETCH_SCORE_THRESHOLD))),
        "prefetch_min_turns": int(os.environ.get("QDRANT_PREFETCH_MIN_TURNS", str(PREFETCH_MIN_TURNS))),
        "prefetch_category_boost": PREFETCH_CATEGORY_BOOST,
        # Search priority filter
        "search_min_priority": int(os.environ.get("QDRANT_SEARCH_MIN_PRIORITY", str(SEARCH_MIN_PRIORITY))),
        # Consolidation + evolution
        "correction_topic_threshold": int(os.environ.get("QDRANT_CORRECTION_TOPIC_THRESHOLD", str(CONSOLIDATION_CORRECTION_TOPIC_THRESHOLD))),
        "orphan_age_days": int(os.environ.get("QDRANT_ORPHAN_AGE_DAYS", str(CONSOLIDATION_ORPHAN_AGE_DAYS))),
        # Session-end extraction
        "session_end_auto_extract": os.environ.get("QDRANT_SESSION_END_EXTRACT", str(SESSION_END_AUTO_EXTRACT)).lower() == "true",
        # Memory lifecycle (v0.9.0)
        "auto_stale_enabled": os.environ.get("QDRANT_AUTO_STALE", str(AUTO_STALE_ENABLED)).lower() == "true",
        "auto_prune_enabled": os.environ.get("QDRANT_AUTO_PRUNE", str(AUTO_PRUNE_ENABLED)).lower() == "true",
        "auto_stale_days": int(os.environ.get("QDRANT_AUTO_STALE_DAYS", str(AUTO_STALE_DAYS))),
        "auto_prune_days": int(os.environ.get("QDRANT_AUTO_PRUNE_DAYS", str(AUTO_PRUNE_DAYS))),
    }

    config_path = get_hermes_home() / "qdrant-memory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            logger.warning("Failed to load config from %s", config_path, exc_info=False)

    # v0.9.0: Config validation — warn and fallback for invalid values
    _validate_config(config)

    return config


def _validate_config(config: dict) -> None:
    """Validate config values, warn on invalid, fall back to defaults."""
    # Float range checks (0.0-1.0)
    for key in ("dedup_threshold", "search_recency_weight", "prefetch_score_threshold"):
        val = config.get(key)
        if val is not None:
            try:
                fval = float(val)
                if not (0.0 <= fval <= 1.0):
                    default = {
                        "dedup_threshold": DEDUP_THRESHOLD,
                        "search_recency_weight": SEARCH_RECENCY_WEIGHT,
                        "prefetch_score_threshold": PREFETCH_SCORE_THRESHOLD,
                    }[key]
                    logger.warning(
                        "Config '%s'=%s out of range [0.0, 1.0], falling back to %s",
                        key, fval, default,
                    )
                    config[key] = default
            except (ValueError, TypeError):
                logger.warning("Config '%s' is not a valid float: %s", key, val)

    # Positive int checks
    for key in ("prefetch_top_k", "prefetch_min_turns", "orphan_age_days",
                "correction_topic_threshold", "auto_stale_days", "auto_prune_days"):
        val = config.get(key)
        if val is not None:
            try:
                ival = int(val)
                if ival < 0:
                    defaults = {
                        "prefetch_top_k": PREFETCH_TOP_K,
                        "prefetch_min_turns": PREFETCH_MIN_TURNS,
                        "orphan_age_days": CONSOLIDATION_ORPHAN_AGE_DAYS,
                        "correction_topic_threshold": CONSOLIDATION_CORRECTION_TOPIC_THRESHOLD,
                        "auto_stale_days": AUTO_STALE_DAYS,
                        "auto_prune_days": AUTO_PRUNE_DAYS,
                    }
                    logger.warning(
                        "Config '%s'=%s is negative, falling back to %s",
                        key, ival, defaults[key],
                    )
                    config[key] = defaults[key]
            except (ValueError, TypeError):
                logger.warning("Config '%s' is not a valid int: %s", key, val)

    # Priority range check (1-5)
    for key in ("search_min_priority",):
        val = config.get(key)
        if val is not None:
            try:
                ival = int(val)
                if not (1 <= ival <= 5):
                    logger.warning(
                        "Config '%s'=%s out of range [1, 5], falling back to %s",
                        key, ival, SEARCH_MIN_PRIORITY,
                    )
                    config[key] = SEARCH_MIN_PRIORITY
            except (ValueError, TypeError):
                logger.warning("Config '%s' is not a valid int: %s", key, val)

    # Collection name not empty
    if not config.get("collection_name", "").strip():
        # Auto-generated is fine, just log it
        logger.debug("collection_name is empty — will be auto-generated")

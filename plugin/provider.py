"""QdrantMemoryProvider — main MemoryProvider implementation.

Wires together QdrantStore, FileIndexer, LearningStore, ConsolidationEngine.
All tool handling lives here. Schema definitions live in schemas.py.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .config import (
    VECTOR_DIM,
    BREAKER_THRESHOLD,
    BREAKER_COOLDOWN_SECS,
    AUTO_SYNC_CONVERSATIONS,
    load_config,
)
from .embeddings import embed
from .schemas import (
    PROFILE_SCHEMA,
    SEARCH_SCHEMA,
    REMEMBER_SCHEMA,
    FORGET_SCHEMA,
    INDEX_SCHEMA,
    CONSOLIDATE_SCHEMA,
)
from .store import QdrantStore
from .indexer import FileIndexer
from .consolidation import ConsolidationEngine

logger = logging.getLogger(__name__)


class QdrantMemoryProvider(MemoryProvider):
    """Qdrant vector memory — local Qdrant + any OpenAI-compatible embedding API."""

    def __init__(self):
        self._config = None
        self._store: Optional[QdrantStore] = None
        self._indexer: Optional[FileIndexer] = None
        self._consolidation: Optional[ConsolidationEngine] = None
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._session_id = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "hermes-memory-qdrant"

    def is_available(self) -> bool:
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            return False
        cfg = load_config()
        return bool(cfg.get("embedding_base_url")) and bool(cfg.get("embedding_api_key"))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = load_config()
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._session_id = session_id
        self._store = QdrantStore(self._config)

        # Sub-components all share the same embed function
        _efn = lambda texts: embed(texts, self._config)

        self._indexer = FileIndexer(self._store, embed_fn=_efn, config=self._config)
        self._consolidation = ConsolidationEngine(
            self._store, embed_fn=_efn, learning_store=None
        )

    def shutdown(self) -> None:
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

    # ── Circuit breaker ───────────────────────────────────────────────────

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + BREAKER_COOLDOWN_SECS
            logger.warning(
                "Qdrant circuit breaker tripped after %d consecutive failures. Pausing for %ds.",
                self._consecutive_failures, BREAKER_COOLDOWN_SECS,
            )

    # ── System prompt ─────────────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        coll = self._store.collection if self._store else "?"
        dedup = self._config.get("dedup_enabled", True) if self._config else True
        return (
            "# Qdrant Memory\n"
            f"Active. Qdrant collection: {coll} (dim={VECTOR_DIM}).\n"
            f"Pre-save dedup: {'ON' if dedup else 'OFF'}. "
            "Same content auto-updates existing entry instead of creating duplicates.\n"
            "Use qdrant_search to find memories, qdrant_remember to store facts, "
            "qdrant_profile for a full overview, qdrant_forget to delete.\n"
            "qdrant_remember also accepts optional 'tags' (string array) for better filtering. "
            "qdrant_search accepts optional 'recency_weight' (0.0-1.0) to favor fresh results."
        )

    # ── Prefetch ──────────────────────────────────────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Qdrant Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open() or not self._store:
            return

        def _run():
            try:
                results = self._store.search(query, top_k=5, user_id=self._user_id)
                if results:
                    lines = [
                        f"- [{r['category']}] {r['memory']} (score: {r['score']})"
                        for r in results
                    ]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Qdrant prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="qdrant-prefetch"
        )
        self._prefetch_thread.start()

    # ── Turn sync ─────────────────────────────────────────────────────────

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._is_breaker_open() or not self._store:
            return

        # Memory hygiene: auto-sync is OFF by default
        if not self._config or not self._config.get("auto_sync_conversations", AUTO_SYNC_CONVERSATIONS):
            return

        def _sync():
            try:
                content = user_content.strip()
                if content and len(content) > 20:
                    self._store.add(
                        content=content[:500],
                        user_id=self._user_id,
                        agent_id=self._agent_id,
                        category="conversation",
                    )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Qdrant sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="qdrant-sync")
        self._sync_thread.start()

    # ── Tools ─────────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            PROFILE_SCHEMA, SEARCH_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA,
            INDEX_SCHEMA,
            CONSOLIDATE_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Qdrant memory temporarily unavailable (multiple consecutive failures)."
            })

        if not self._store:
            return tool_error("Qdrant store not initialized")

        try:
            # ── Core tools ──────────────────────────────────────────────
            if tool_name == "qdrant_profile":
                memories = self._store.get_all(user_id=self._user_id)
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = []
                for m in memories:
                    extra = ""
                    if m.get("tags"):
                        extra += f" [{', '.join(m['tags'])}]"
                    if m.get("version", 1) > 1:
                        extra += f" (v{m['version']})"
                    lines.append(f"[{m['category']}]{extra} {m['memory']}")
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})

            elif tool_name == "qdrant_search":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                top_k = min(int(args.get("top_k", 10)), 50)
                recency_weight = args.get("recency_weight", None)
                if recency_weight is not None:
                    recency_weight = float(recency_weight)
                tags = args.get("tags", None)
                results = self._store.search(
                    query, top_k=top_k, user_id=self._user_id,
                    recency_weight=recency_weight, tags=tags,
                )
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [
                    {
                        "memory": r["memory"],
                        "score": r["score"],
                        "category": r.get("category", ""),
                        "tags": r.get("tags", []),
                        "version": r.get("version", 1),
                    }
                    for r in results
                ]
                return json.dumps({"results": items, "count": len(items)})

            elif tool_name == "qdrant_remember":
                content = args.get("content", "")
                if not content:
                    return tool_error("Missing required parameter: content")
                category = args.get("category", "fact")
                tags = args.get("tags", None)
                point_id = self._store.add(
                    content=content, user_id=self._user_id,
                    agent_id=self._agent_id, category=category,
                    tags=tags,
                )
                self._record_success()
                return json.dumps({"result": "Memory stored.", "id": point_id})

            elif tool_name == "qdrant_forget":
                point_id = args.get("point_id", "")
                if not point_id:
                    return tool_error("Missing required parameter: point_id")
                dry_run = args.get("dry_run", True)
                if dry_run:
                    memory = self._store.get_point(point_id)
                    if memory:
                        return json.dumps({
                            "dry_run": True,
                            "result": f"Would delete: [{memory.get('category', '?')}] {memory.get('content', '?')[:120]}",
                            "point_id": point_id,
                            "category": memory.get("category", ""),
                            "created_at": memory.get("created_at", ""),
                        })
                    return json.dumps({"dry_run": True, "result": "Memory not found."})
                ok = self._store.delete(point_id)
                self._record_success()
                if ok:
                    return json.dumps({"result": "Memory deleted.", "id": point_id})
                return json.dumps({"result": "Memory not found or already deleted."})

            # ── Indexing ────────────────────────────────────────────────
            elif tool_name == "qdrant_index":
                if not self._indexer:
                    return tool_error("Indexer not initialized")
                paths = args.get("paths", [])
                if not paths:
                    return tool_error("Missing required parameter: paths")
                result = self._indexer.index(
                    paths=paths,
                    dry_run=args.get("dry_run", True),
                    max_files=args.get("max_files"),
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                )
                self._record_success()
                return json.dumps(result)

            # ── Consolidation ───────────────────────────────────────────
            elif tool_name == "qdrant_consolidate":
                if not self._consolidation:
                    return tool_error("Consolidation engine not initialized")
                result = self._consolidation.consolidate(
                    scope=args.get("scope", "memory"),
                    max_points=int(args.get("max_points", 500)),
                    max_groups=int(args.get("max_groups", 20)),
                    include_examples=bool(args.get("include_examples", False)),
                )
                self._record_success()
                return json.dumps(result)

            return tool_error(f"Unknown tool: {tool_name}")

        except Exception as e:
            self._record_failure()
            return tool_error(f"Qdrant memory error: {e}")

    # ── Config schema (for hermes memory setup wizard) ────────────────────

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "qdrant_url", "description": "Qdrant server URL", "default": "http://localhost:6333"},
            {"key": "qdrant_api_key", "description": "Qdrant API key (optional)", "secret": True, "env_var": "QDRANT_API_KEY"},
            {"key": "embedding_base_url", "description": "OpenAI-compatible embedding endpoint URL", "required": True, "env_var": "EMBEDDING_BASE_URL"},
            {"key": "embedding_api_key", "description": "API key for the embedding service", "secret": True, "required": True, "env_var": "EMBEDDING_API_KEY"},
            {"key": "embedding_model", "description": "Embedding model name", "default": "doubao-embedding-vision"},
            {"key": "collection_name", "description": "Unique collection name (auto-generated if empty). MUST be unique per deployment.", "env_var": "QDRANT_COLLECTION"},
            # Memory hygiene settings
            {"key": "dedup_threshold", "description": "Cosine similarity threshold for pre-save dedup (0.0-1.0, default: 0.85)", "default": "0.85", "env_var": "QDRANT_DEDUP_THRESHOLD"},
            {"key": "dedup_enabled", "description": "Enable pre-save dedup (default: true)", "default": "true", "env_var": "QDRANT_DEDUP_ENABLED"},
            {"key": "auto_sync_conversations", "description": "Auto-save user messages to memory (default: false)", "default": "false", "env_var": "QDRANT_AUTO_SYNC"},
            {"key": "search_recency_weight", "description": "How much to favor recency in search results (0.0-1.0, default: 0.0)", "default": "0.0", "env_var": "QDRANT_RECENCY_WEIGHT"},
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        from pathlib import Path

        config_path = Path(hermes_home) / "qdrant-memory.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        secrets_map = {"qdrant_api_key": "QDRANT_API_KEY", "embedding_api_key": "EMBEDDING_API_KEY"}
        secrets_to_env = {}
        non_secrets = dict(existing)

        for key, env_var in secrets_map.items():
            if key in values and values[key]:
                secrets_to_env[env_var] = values.pop(key, "")
                non_secrets.pop(key, None)

        non_secrets.update({k: v for k, v in values.items() if v is not None and v != ""})
        if non_secrets:
            config_path.write_text(json.dumps(non_secrets, indent=2))
        elif config_path.exists():
            config_path.unlink()

        if secrets_to_env:
            env_path = Path(hermes_home) / ".env"
            env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
            for k, v in secrets_to_env.items():
                if v:
                    line = f"{k}={v}"
                    import re
                    if re.search(rf"^{re.escape(k)}=", env_text, re.MULTILINE):
                        env_text = re.sub(rf"^{re.escape(k)}=.*", line, env_text, flags=re.MULTILINE)
                    else:
                        env_text += f"\n{line}"
            env_path.write_text(env_text.strip() + "\n", encoding="utf-8")

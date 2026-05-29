"""QdrantMemoryProvider — main MemoryProvider implementation.

Wires together QdrantStore, FileIndexer, LearningStore, ConsolidationEngine.
All tool handling lives here. Schema definitions live in schemas.py.

v0.7.0: backfill tool, min_priority search, evolved_from remember, quick consolidate
v0.8.0: topics tool, session-end auto-extract, smart prefetch (skip short convos)
v0.9.0: cross-session context bridge, auto-stale/prune
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .config import (
    VECTOR_DIM,
    BREAKER_THRESHOLD,
    BREAKER_COOLDOWN_SECS,
    AUTO_SYNC_CONVERSATIONS,
    PREFETCH_TOP_K,
    PREFETCH_SCORE_THRESHOLD,
    PREFETCH_MIN_TURNS,
    SEARCH_MIN_PRIORITY,
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
    BACKFILL_SCHEMA,
    TOPICS_SCHEMA,
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

        # Register hooks with the REAL PluginManager (not the _ProviderCollector
        # stub that silently no-ops register_hook).  The memory provider loading
        # path uses _ProviderCollector which swallows hook registration, so we
        # must do it here during initialize() when the real PluginManager exists.
        self._register_hooks_with_plugin_manager()

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

    def _register_hooks_with_plugin_manager(self) -> None:
        """Register lifecycle hooks with the real Hermes PluginManager.

        The memory provider loading path (plugins/memory/__init__.py)
        uses _ProviderCollector, whose register_hook() is a no-op.
        This method bypasses that by reaching the real PluginManager's
        internal _hooks dict directly.
        """
        try:
            from hermes_cli.plugins import get_plugin_manager
            pm = get_plugin_manager()

            # Dedup guard: Hermes may instantiate QdrantMemoryProvider multiple
            # times per session (memory provider path + plugin tools path), each
            # calling initialize().  Without this check, hooks accumulate and
            # fire N times per turn (double embedding cost, double sync).
            _to_register = {
                "pre_llm_call": self.on_pre_llm_call,
                "post_llm_call": self.on_post_llm_call,
                "on_session_end": self.on_session_end_hook,
            }
            _registered = 0
            for hook_name, handler in _to_register.items():
                existing = pm._hooks.get(hook_name, [])
                if handler not in existing:
                    existing.append(handler)
                    _registered += 1

            if _registered:
                logger.info(
                    "Qdrant: registered %d hooks with PluginManager "
                    "(pre_llm_call, post_llm_call, on_session_end)",
                    _registered,
                )
            else:
                logger.debug("Qdrant: hooks already registered — skipping duplicate")
        except Exception as e:
            logger.warning("Qdrant: failed to register hooks with PluginManager: %s", e)

    # ── System prompt ─────────────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        coll = self._store.collection if self._store else "?"
        dedup = self._config.get("dedup_enabled", True) if self._config else True

        # Read version from VERSION file
        version = "?.?"
        try:
            from pathlib import Path
            vfile = Path(__file__).parent / "VERSION"
            if vfile.exists():
                version = vfile.read_text(encoding="utf-8").strip().lstrip("v")
        except Exception:
            pass

        # Memory stats
        stats = ""
        try:
            if self._store:
                all_mems = self._store.get_all(user_id=self._user_id, limit=0)
                total = len(all_mems)
                cats = {}
                corrections = 0
                for m in all_mems:
                    cat = m.get("category", "unknown")
                    cats[cat] = cats.get(cat, 0) + 1
                    if cat == "correction":
                        corrections += 1
                cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
                stats = f"\nStats: {total} memories ({cat_str}). Corrections tracked: {corrections}."
        except Exception:
            pass

        return (
            f"# Qdrant Memory (Self-Healing v{version})\n"
            f"Active. Collection: {coll} (dim={VECTOR_DIM}).{stats}\n"
            f"Dedup: {'ON' if dedup else 'OFF'}. Auto-context: ON (top {PREFETCH_TOP_K}).\n"
            "Categories: preference, fact, decision, goal, instruction, correction.\n"
            "Use qdrant_search to find memories, qdrant_remember to store facts, "
            "qdrant_profile for a full overview, qdrant_forget to delete.\n"
            "qdrant_remember accepts: category (use 'correction' for behavior fixes), "
            "priority (1-5), origin ('user_correction'|'agent_discovery'|'explicit'|'auto'), "
            "tags (string array for filtering), evolved_from (ID of source memory).\n"
            "qdrant_search accepts: recency_weight (0.0-1.0), tags (AND filter), "
            "min_priority (1=all, 5=highest quality only).\n"
            "qdrant_consolidate accepts: quick (skip dedup), auto_stale, auto_prune.\n"
            "qdrant_backfill: backfill missing fields on old memories.\n"
            "qdrant_topics: discover topic clusters in your memories.\n"
            "IMPORTANT: When user corrects your behavior, store it as category='correction' "
            "with priority=1 so it surfaces first in future turns."
        )

    # ── Prefetch (memory_manager path — legacy) ──────────────────────────
    #
    # Hermes memory_manager calls provider.prefetch() at line 348.
    # The hook path (on_pre_llm_call) does its own independent search.
    # Both paths coexist — belt-and-suspenders dispatch.

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

        # v0.8.0: Smart prefetch — skip short conversations
        # (Don't count query length — we don't have conversation history here)
        # The on_pre_llm_call hook handles the full smart-prefetch logic

        top_k = int(self._config.get("prefetch_top_k", PREFETCH_TOP_K)) if self._config else PREFETCH_TOP_K
        score_threshold = float(self._config.get("prefetch_score_threshold", PREFETCH_SCORE_THRESHOLD)) if self._config else PREFETCH_SCORE_THRESHOLD

        def _run():
            try:
                results = self._store.search(
                    query, top_k=top_k, user_id=self._user_id,
                    score_threshold=score_threshold,
                    category_boost={"conversation": 0.0, "correction": 1.3, "preference": 1.2, "instruction": 1.1},
                )
                if results:
                    # Post-filter: exclude conversation, apply min score
                    MIN_SCORE = 0.2
                    filtered = [
                        r for r in results
                        if r.get("category") != "conversation"
                        and r["score"] >= MIN_SCORE
                    ]
                    if filtered:
                        lines = []
                        for r in filtered[:top_k]:
                            cat = r.get("category", "")
                            pri = r.get("priority", 3)
                            origin = r.get("origin", "")
                            # Compact format: [category/priority] content (score)
                            meta = f"{cat}"
                            if pri <= 2:
                                meta = f"{meta}/P{pri}"
                            if origin == "user_correction":
                                meta = f"⚠️{meta}"
                            evolved = ""
                            if r.get("evolved_from"):
                                evolved = f" [evolved from {r['evolved_from'][:8]}]"
                            lines.append(
                                f"- [{meta}] {r['memory']}{evolved} (score: {r['score']})"
                            )
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
            BACKFILL_SCHEMA,
            TOPICS_SCHEMA,
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
                # v0.7.0: min_priority filter
                min_priority = int(args.get("min_priority", 1))
                results = self._store.search(
                    query, top_k=top_k, user_id=self._user_id,
                    recency_weight=recency_weight, tags=tags,
                    min_priority=min_priority,
                )
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = []
                for r in results:
                    item = {
                        "memory": r["memory"],
                        "score": r["score"],
                        "category": r.get("category", ""),
                        "tags": r.get("tags", []),
                        "version": r.get("version", 1),
                        "priority": r.get("priority", 3),
                    }
                    # v0.7.0: Include evolved_from if present
                    if r.get("evolved_from"):
                        item["evolved_from"] = r["evolved_from"]
                    items.append(item)
                return json.dumps({"results": items, "count": len(items)})

            elif tool_name == "qdrant_remember":
                content = args.get("content", "")
                if not content:
                    return tool_error("Missing required parameter: content")
                category = args.get("category", "fact")
                tags = args.get("tags", None)
                priority = min(max(int(args.get("priority", 3)), 1), 5)
                origin = args.get("origin", "explicit")
                # v0.7.0: evolved_from parameter
                evolved_from = args.get("evolved_from", None)
                point_id = self._store.add(
                    content=content, user_id=self._user_id,
                    agent_id=self._agent_id, category=category,
                    tags=tags, priority=priority, origin=origin,
                    evolved_from=evolved_from,
                )
                self._record_success()
                result = {
                    "result": "Memory stored.",
                    "id": point_id,
                    "priority": priority,
                    "category": category,
                    "origin": origin,
                }
                if evolved_from:
                    result["evolved_from"] = evolved_from
                return json.dumps(result)

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
                # v0.9.0: auto_stale and auto_prune require config-level enable
                auto_stale = args.get("auto_stale", False)
                auto_prune = args.get("auto_prune", False)
                if auto_stale and not self._config.get("auto_stale_enabled", False):
                    auto_stale = False
                    logger.info("auto_stale requested but QDRANT_AUTO_STALE not enabled")
                if auto_prune and not self._config.get("auto_prune_enabled", False):
                    auto_prune = False
                    logger.info("auto_prune requested but QDRANT_AUTO_PRUNE not enabled")
                result = self._consolidation.consolidate(
                    scope=args.get("scope", "memory"),
                    max_points=int(args.get("max_points", 500)),
                    max_groups=int(args.get("max_groups", 20)),
                    include_examples=bool(args.get("include_examples", False)),
                    quick=bool(args.get("quick", False)),
                    auto_stale=auto_stale,
                    auto_prune=auto_prune,
                )
                self._record_success()
                return json.dumps(result)

            # ── Backfill (v0.7.0) ──────────────────────────────────────
            elif tool_name == "qdrant_backfill":
                defaults = args.get("defaults", {})
                if not defaults:
                    return tool_error("Missing required parameter: defaults")
                dry_run = args.get("dry_run", True)
                result = self._store.backfill_fields(
                    defaults=defaults, dry_run=dry_run,
                )
                self._record_success()
                return json.dumps(result)

            # ── Topics (v0.8.0) ────────────────────────────────────────
            elif tool_name == "qdrant_topics":
                if not self._consolidation:
                    return tool_error("Consolidation engine not initialized")
                min_cluster_size = int(args.get("min_cluster_size", 2))
                similarity_threshold = float(args.get("similarity_threshold", 0.75))
                # Fetch points for clustering
                points = self._store.get_all(user_id=self._user_id, limit=500)
                if not points:
                    return json.dumps({"result": "No memories to cluster.", "clusters": []})
                # Need vectors for clustering — re-fetch with vectors
                try:
                    from .clustering import TopicClustering
                    tc = TopicClustering(
                        similarity_threshold=similarity_threshold,
                        min_cluster_size=min_cluster_size,
                    )
                    # Fetch with vectors via consolidation engine
                    all_points = self._consolidation._fetch_all_points(
                        self._store.collection, 500
                    )
                    clusters = tc.find_clusters(all_points)
                    self._record_success()
                    if not clusters:
                        return json.dumps({
                            "result": "No topic clusters found.",
                            "clusters": [],
                            "points_analyzed": len(all_points),
                        })
                    # Format for output
                    output_clusters = []
                    for c in clusters:
                        output_clusters.append({
                            "topic": c["topic_label"],
                            "size": c["size"],
                            "avg_similarity": c["avg_similarity"],
                            "categories": c.get("categories", []),
                            "content_previews": c.get("content_previews", []),
                        })
                    return json.dumps({
                        "result": f"Found {len(clusters)} topic clusters.",
                        "clusters": output_clusters,
                        "points_analyzed": len(all_points),
                    })
                except Exception as e:
                    self._record_failure()
                    return tool_error(f"Topic clustering failed: {e}")

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
            # v0.7.0+ new config
            {"key": "prefetch_min_turns", "description": "Min user turns before prefetch starts (default: 3)", "default": "3", "env_var": "QDRANT_PREFETCH_MIN_TURNS"},
            {"key": "search_min_priority", "description": "Min priority for search results (1=all, 5=highest quality, default: 1)", "default": "1", "env_var": "QDRANT_SEARCH_MIN_PRIORITY"},
            # v0.9.0 lifecycle
            {"key": "auto_stale_enabled", "description": "Enable auto-stale (bump old low-priority to 5, default: false)", "default": "false", "env_var": "QDRANT_AUTO_STALE"},
            {"key": "auto_prune_enabled", "description": "Enable auto-prune (DELETE old priority-5 memories, default: false)", "default": "false", "env_var": "QDRANT_AUTO_PRUNE"},
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
                    if re.search(rf"^{re.escape(k)}=", env_text, re.MULTILINE):
                        env_text = re.sub(rf"^{re.escape(k)}=.*", line, env_text, flags=re.MULTILINE)
                    else:
                        env_text += f"\n{line}"
            env_path.write_text(env_text.strip() + "\n", encoding="utf-8")

    # ── Optional hooks (self-healing features) ──────────────────────────

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to Qdrant.

        When the built-in memory tool writes (add/replace/remove), this hook
        fires so Qdrant stays in sync. Useful for ensuring critical rules
        written to local memory are also available for semantic search.
        """
        if self._is_breaker_open() or not self._store:
            return

        try:
            if action == "add":
                self._store.add(
                    content=content,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    category="instruction",
                    priority=2,
                    origin="auto",
                    tags=["local-memory-mirror", target],
                )
            elif action == "replace":
                # Dedup handles this — same content updates existing point
                self._store.add(
                    content=content,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    category="instruction",
                    priority=2,
                    origin="auto",
                    tags=["local-memory-mirror", target],
                )
            # "remove" — don't auto-delete from Qdrant (let consolidation handle it)
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.debug("on_memory_write mirror failed: %s", e)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Session-end hook for memory provider interface.

        v0.8.0: Smart auto-extraction of key facts from conversation.
        Only extracts substantive user messages that contain facts/preferences/instructions.
        Skips questions, tool outputs, and short messages.
        Auto-extracted memories get category='auto_extract', priority=4.
        """
        if not self._config or not self._config.get("session_end_auto_extract", True):
            return
        if self._is_breaker_open() or not self._store:
            return

        try:
            extracted = 0
            user_messages = [
                m for m in messages
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ]

            for msg in user_messages:
                content = msg["content"].strip()
                if len(content) < 50:
                    continue  # Skip short messages

                # Skip questions
                if content.endswith("?") and content.count("?") >= 1:
                    if len(content) < 100:  # Short questions are definitely not facts
                        continue

                # Skip tool output (JSON-like)
                if content.startswith("{") and content.endswith("}"):
                    continue
                if content.startswith("[") and content.endswith("]"):
                    continue

                # Quick dedup check — search existing memories
                try:
                    existing = self._store.search(
                        content[:200],
                        top_k=3,
                        user_id=self._user_id,
                        score_threshold=0.85,
                    )
                    if existing:
                        continue  # Already known
                except Exception:
                    pass

                # Store as auto-extracted memory
                try:
                    self._store.add(
                        content=content[:500],
                        user_id=self._user_id,
                        agent_id=self._agent_id,
                        category="auto_extract",
                        priority=4,
                        origin="auto",
                        tags=["session-extract", self._session_id[:8]] if self._session_id else ["session-extract"],
                    )
                    extracted += 1
                except Exception:
                    pass

            if extracted > 0:
                logger.info(
                    "Session-end auto-extracted %d memories from %d user messages",
                    extracted, len(user_messages),
                )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.debug("Session-end auto-extract failed: %s", e)

    # ── Hermes hook adapters (v0.6.1+) ──────────────────────────────────
    #
    # These bridge the Hermes plugin hook system (invoke_hook) to the
    # Qdrant memory provider's internal methods.

    def on_pre_llm_call(self, *, session_id: str = "",
                        user_message=None, conversation_history=None,
                        is_first_turn: bool = False, model: str = "",
                        platform: str = "", sender_id: str = "",
                        **_kwargs) -> dict:
        """pre_llm_call hook: return high-value memories for context injection.

        v0.8.0: Smart prefetch — skip embedding search on short conversations.
        v0.9.0: Cross-session context bridge — inject session starter on first turn.
        """
        logger.info(
            "Qdrant pre_llm_call hook: called (store=%s, breaker=%s, msg_len=%d, first_turn=%s)",
            bool(self._store),
            self._is_breaker_open(),
            len(user_message) if isinstance(user_message, str) else 0,
            is_first_turn,
        )

        if self._is_breaker_open() or not self._store:
            return {}

        query = user_message if isinstance(user_message, str) else ""
        if not query and not is_first_turn:
            return {}

        # v0.9.0: Cross-session context bridge — first turn gets starter context
        if is_first_turn:
            return self._session_starter_context()

        # v0.8.0: Smart prefetch — skip short conversations
        min_turns = int(self._config.get("prefetch_min_turns", PREFETCH_MIN_TURNS)) if self._config else PREFETCH_MIN_TURNS
        if conversation_history and min_turns > 0:
            user_turns = len([
                m for m in conversation_history
                if m.get("role") == "user"
            ])
            if user_turns < min_turns:
                logger.info(
                    "Qdrant pre_llm_call: skipping prefetch (user_turns=%d < min_turns=%d)",
                    user_turns, min_turns,
                )
                return {}

        try:
            top_k = int(self._config.get("prefetch_top_k", PREFETCH_TOP_K)) if self._config else PREFETCH_TOP_K
            # Fetch extra so category filtering doesn't starve results
            results = self._store.search(
                query, top_k=top_k * 3, user_id=self._user_id,
                score_threshold=0.0,  # raw pass from Qdrant — we filter below
                category_boost={
                    "conversation": 0.0,  # zero = exclude entirely
                    "correction": 1.3,
                    "preference": 1.2,
                    "instruction": 1.1,
                },
            )
            if not results:
                return {}

            # Post-filter: exclude conversation, apply min score
            MIN_SCORE = 0.2
            filtered = [
                r for r in results
                if r.get("category") != "conversation"
                and r.get("category") != "auto_extract"  # v0.8.0: don't surface auto-extracts in prefetch
                and r["score"] >= MIN_SCORE
            ]

            if not filtered:
                logger.info("Qdrant pre_llm_call: all results filtered out (only conversation/low-score)")
                return {}

            lines = []
            for r in filtered[:top_k]:
                cat = r.get("category", "")
                pri = r.get("priority", 3)
                origin = r.get("origin", "")
                meta = f"{cat}"
                if pri <= 2:
                    meta = f"{meta}/P{pri}"
                if origin == "user_correction":
                    meta = f"⚠️{meta}"
                evolved = ""
                if r.get("evolved_from"):
                    evolved = f" [evolved]"
                lines.append(
                    f"- [{meta}] {r['memory']}{evolved} (score: {r['score']})"
                )
            context = "## Qdrant Memory\n" + "\n".join(lines)
            logger.info(
                "Qdrant pre_llm_call: injected %d results (%d chars, filtered %d)",
                len(filtered), len(context),
                len(results) - len(filtered),
            )
            return {"context": context}
        except Exception as e:
            logger.debug("Qdrant pre_llm_call direct search failed: %s", e)
            return {}

    def _session_starter_context(self) -> dict:
        """v0.9.0: Cross-session context bridge.

        On first turn, inject session starter context:
        - Top highest-priority memories (corrections > instructions > preferences)
        - Recent memories from last 24 hours
        """
        try:
            top_k = int(self._config.get("prefetch_top_k", PREFETCH_TOP_K)) if self._config else PREFETCH_TOP_K

            # Get top-priority memories using empty-ish query and priority sorting
            # We fetch from get_all and manually sort by priority
            all_mems = self._store.get_all(user_id=self._user_id, limit=100)

            if not all_mems:
                return {}

            # Sort by priority (1=highest first), exclude conversation and auto_extract
            filtered = [
                m for m in all_mems
                if m.get("category") not in ("conversation", "auto_extract", "topic_summary")
            ]
            filtered.sort(key=lambda m: (
                {"correction": 0, "instruction": 1, "preference": 2, "decision": 3,
                 "fact": 4, "goal": 5}.get(m.get("category", ""), 6),
                m.get("priority", 3),
            ))

            # Take top N
            top_mems = filtered[:top_k]

            # Also get recent memories from last 24 hours
            now = datetime.now(timezone.utc)
            yesterday = now - timedelta(hours=24)
            recent = []
            for m in filtered:
                updated = m.get("updated_at") or m.get("created_at", "")
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt > yesterday and m["id"] not in {x["id"] for x in top_mems}:
                            recent.append(m)
                    except (ValueError, TypeError):
                        pass
            recent = recent[:3]

            if not top_mems and not recent:
                return {}

            lines = ["## Session Context"]
            if top_mems:
                lines.append(f"\n### Priority Memories ({len(top_mems)})")
                for m in top_mems:
                    cat = m.get("category", "")
                    pri = m.get("priority", 3)
                    origin = m.get("origin", "")
                    meta = f"{cat}"
                    if pri <= 2:
                        meta = f"{meta}/P{pri}"
                    if origin == "user_correction":
                        meta = f"⚠️{meta}"
                    lines.append(f"- [{meta}] {m['memory']}")

            if recent:
                lines.append(f"\n### Recent (24h)")
                for m in recent:
                    lines.append(f"- [{m.get('category', '')}] {m['memory']}")

            context = "\n".join(lines)
            logger.info(
                "Qdrant session starter: injected %d priority + %d recent memories",
                len(top_mems), len(recent),
            )
            return {"context": context}
        except Exception as e:
            logger.debug("Qdrant session starter context failed: %s", e)
            return {}

    def on_post_llm_call(self, *, session_id: str = "",
                         user_message=None, assistant_response=None,
                         conversation_history=None, model: str = "",
                         platform: str = "", **_kwargs) -> None:
        """post_llm_call hook: sync completed turn to Qdrant.

        Persists the user-assistant exchange and queues a prefetch
        for the next turn.
        """
        user_content = user_message if isinstance(user_message, str) else ""
        assistant_content = assistant_response if isinstance(assistant_response, str) else ""

        if not user_content or not assistant_content:
            return

        try:
            self.sync_turn(user_content, assistant_content, session_id=session_id)
            self.queue_prefetch(user_content, session_id=session_id)
            logger.info("Qdrant post_llm_call hook: synced turn (user=%d chars)", len(user_content))
        except Exception as e:
            logger.debug("Qdrant post_llm_call hook failed: %s", e)

    def on_session_end_hook(self, *, session_id: str = "",
                            completed: bool = False, interrupted: bool = False,
                            messages: list | None = None,
                            model: str = "", platform: str = "",
                            **_kwargs) -> None:
        """on_session_end hook: auto-extract key facts + cleanup.

        v0.8.0: Auto-extracts substantive user messages as memories.
        v0.6.1: Disabled storage of "Session ended: <id>" markers.
        """
        if interrupted or not session_id:
            return

        # v0.8.0: Auto-extract from messages if provided
        if messages and self._config and self._config.get("session_end_auto_extract", True):
            self.on_session_end(messages)

        logger.info("Qdrant on_session_end hook: session %s completed", session_id)

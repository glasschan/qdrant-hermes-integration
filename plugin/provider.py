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
    PREFETCH_TOP_K,
    PREFETCH_SCORE_THRESHOLD,
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
            "tags (string array for filtering).\n"
            "qdrant_search accepts: recency_weight (0.0-1.0), tags (AND filter).\n"
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
                            lines.append(
                                f"- [{meta}] {r['memory']} (score: {r['score']})"
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
                priority = min(max(int(args.get("priority", 3)), 1), 5)
                origin = args.get("origin", "explicit")
                point_id = self._store.add(
                    content=content, user_id=self._user_id,
                    agent_id=self._agent_id, category=category,
                    tags=tags, priority=priority, origin=origin,
                )
                self._record_success()
                return json.dumps({"result": "Memory stored.", "id": point_id, "priority": priority, "category": category, "origin": origin})

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

        v0.6.1: Disabled auto-extraction of "Session topics" summaries.
        These were polluting search results with low-value conversation
        fragments that matched semantically but carried no useful signal.
        Real valuable memories are stored via explicit qdrant_remember
        calls (corrections, preferences, decisions).
        """
        pass

    # ── Hermes hook adapters (v0.6.1) ──────────────────────────────────
    #
    # These bridge the Hermes plugin hook system (invoke_hook) to the
    # Qdrant memory provider's internal methods.  The hook path runs
    # alongside the memory_manager path — belt-and-suspenders so that
    # if the memory_manager dispatch silently fails, the hook dispatch
    # still fires.
    #
    # Hook signatures must match what conversation_loop.py passes via
    # invoke_hook(**kwargs).  Unknown kwargs are absorbed by **_kwargs.

    def on_pre_llm_call(self, *, session_id: str = "",
                        user_message=None, conversation_history=None,
                        is_first_turn: bool = False, model: str = "",
                        platform: str = "", sender_id: str = "",
                        **_kwargs) -> dict:
        """pre_llm_call hook: return high-value memories for context injection.

        v0.6.1: Completely independent of prefetch cache. Does a fresh
        synchronous search and explicitly EXCLUDES conversation category
        (auto-generated session summaries are noise, not memory).
        """
        logger.info(
            "Qdrant pre_llm_call hook: called (store=%s, breaker=%s, msg_len=%d)",
            bool(self._store),
            self._is_breaker_open(),
            len(user_message) if isinstance(user_message, str) else 0,
        )

        if self._is_breaker_open() or not self._store:
            return {}

        query = user_message if isinstance(user_message, str) else ""
        if not query:
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
                lines.append(
                    f"- [{meta}] {r['memory']} (score: {r['score']})"
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
                            model: str = "", platform: str = "",
                            **_kwargs) -> None:
        """on_session_end hook: cleanup only, no data storage.

        v0.6.1: Disabled storage of "Session ended: <id>" markers.
        These were polluting search results. Session lifecycle is
        tracked by the session DB, not Qdrant.
        """
        if interrupted or not session_id:
            return
        logger.info("Qdrant on_session_end hook: session %s completed", session_id)

"""Qdrant vector memory plugin — MemoryProvider interface.

Semantic long-term memory via local Qdrant + any OpenAI-compatible embedding API.

Env vars:
  QDRANT_URL          — Qdrant server URL (default: http://localhost:6333)
  QDRANT_API_KEY      — Qdrant API key (optional)
  EMBEDDING_BASE_URL  — OpenAI-compatible embeddings endpoint (required)
  EMBEDDING_API_KEY   — API key for the embedding service (required)
  EMBEDDING_MODEL     — Embedding model name (default: doubao-embedding-vision)

Or via $HERMES_HOME/qdrant-memory.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from plugin.indexer import FileIndexer
from plugin.learning import LearningStore
from plugin.consolidation import ConsolidationEngine

logger = logging.getLogger(__name__)

VECTOR_DIM = 2048

# Circuit breaker
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
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
    }

    config_path = get_hermes_home() / "qdrant-memory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed(texts: list[str], config: dict) -> list[list[float]]:
    """Get embeddings from an OpenAI-compatible API."""
    import requests

    base = config["embedding_base_url"].rstrip("/")
    url = urljoin(base + "/", "embeddings")

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config['embedding_api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": config["embedding_model"],
            "input": texts,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    # Sort by index to preserve ordering
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "qdrant_profile",
    "description": (
        "Retrieve all stored vector memories about the user — preferences, facts, "
        "project context. Returns everything from Qdrant vector store."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "qdrant_search",
    "description": (
        "Search vector memories by semantic meaning. Uses Qdrant + embeddings "
        "to find the most relevant stored facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for semantically."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "qdrant_remember",
    "description": (
        "Store a durable fact about the user in Qdrant vector memory. "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "category": {
                "type": "string",
                "enum": ["preference", "fact", "decision", "goal", "instruction"],
                "description": "Category (default: fact).",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "qdrant_forget",
    "description": (
        "Delete a vector memory by its point ID. "
        "Dry-run defaults to true — always preview first before live deletion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "The point ID (UUID) to delete."},
            "dry_run": {
                "type": "boolean",
                "description": "When true (default), only report what would be deleted without deleting.",
            },
        },
        "required": ["point_id"],
    },
}

INDEX_SCHEMA = {
    "name": "qdrant_index",
    "description": (
        "Safely index markdown/text files or directories into Qdrant memory. "
        "Dry-run defaults to true — always preview first. "
        "Supports manifest sync: detects changed and deleted files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files or directories to index.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "When true (default), preview chunks without upserting.",
            },
            "max_files": {
                "type": "integer",
                "description": "Max files to scan (default: 500).",
                "minimum": 1,
            },
        },
        "required": ["paths"],
    },
}

CONSOLIDATE_SCHEMA = {
    "name": "qdrant_consolidate",
    "description": (
        "Generate a read-only memory consolidation report. "
        "Finds duplicates, stale memories, and quality warnings. "
        "NEVER mutates data — report-only by design."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": "Which collections to scan: memory, learning, or both.",
                "enum": ["memory", "learning", "both"],
            },
            "max_points": {
                "type": "integer",
                "description": "Max points to scan (default: 500).",
                "minimum": 10,
            },
            "max_groups": {
                "type": "integer",
                "description": "Max duplicate groups to return (default: 20).",
                "minimum": 1,
            },
            "include_examples": {
                "type": "boolean",
                "description": "Include redacted content examples in report.",
            },
        },
        "required": [],
    },
}

LEARNING_STORE_SCHEMA = {
    "name": "qdrant_learning_store",
    "description": (
        "Store an explicit procedural learning in the separate Qdrant learning collection. "
        "Manual/gated only — not automatic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "lesson": {"type": "string", "description": "The durable lesson/procedure learned."},
            "learning_type": {
                "type": "string",
                "description": "tool_failure_lesson, user_correction, workflow_lesson, or environment_quirk.",
            },
            "trigger": {"type": "string", "description": "Situation that should trigger recall."},
            "mistake": {"type": "string", "description": "What went wrong."},
            "correction": {"type": "string", "description": "The corrected action."},
            "evidence": {"type": "string", "description": "Evidence supporting the lesson."},
            "tool_name": {"type": "string", "description": "Tool involved, if any."},
            "importance": {"type": "integer", "description": "Importance 1-10 (default: 7).", "minimum": 1, "maximum": 10},
            "confidence": {"type": "number", "description": "Confidence 0-1 (default: 0.8).", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
            "promote_to_skill": {"type": "boolean", "description": "Mark as skill promotion candidate."},
        },
        "required": ["lesson"],
    },
}

LEARNING_SEARCH_SCHEMA = {
    "name": "qdrant_learning_search",
    "description": "Search procedural learnings from the separate Qdrant learning collection.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 5, max: 20)."},
            "learning_type": {"type": "string", "description": "Optional filter by learning_type."},
        },
        "required": ["query"],
    },
}

LEARNING_PREVIEW_SCHEMA = {
    "name": "qdrant_learning_preview",
    "description": "Preview pending gated learning candidates. Dry-run only — never writes.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

LEARNING_APPROVE_SCHEMA = {
    "name": "qdrant_learning_approve",
    "description": "Approve a pending learning candidate by ID and store it.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "description": "Candidate ID to approve."},
            "dry_run": {"type": "boolean", "description": "When true, preview without storing."},
        },
        "required": ["candidate_id"],
    },
}


# ---------------------------------------------------------------------------
# Qdrant client wrapper
# ---------------------------------------------------------------------------

class _QdrantStore:
    """Thin wrapper around qdrant-client."""

    def __init__(self, config: dict):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._models = models
        url = config["qdrant_url"]
        api_key = config.get("qdrant_api_key") or None
        self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._config = config
        self._collection = self._ensure_collection()

    def _ensure_collection(self) -> str:
        """Use or create OUR collection. NEVER touch another agent's collection.

        - Collection name is set via QDRANT_COLLECTION env var.
        - If empty, auto-generates: hermes_memories_<hostname>_<profile>
        - If already exists, use it (it's ours from a previous session).
        - If not exists, create it.
        - **Never delete any collection.**
        - **Only operate on self._collection — never reference external names.**
        """
        import socket
        from hermes_constants import get_hermes_home

        name = self._config.get("collection_name", "").strip()
        if not name:
            profile = os.path.basename(str(get_hermes_home()))
            hostname = socket.gethostname().split(".")[0]
            name = f"hermes_memories_{hostname}_{profile}"
            logger.info(
                "QDRANT_COLLECTION not set — using auto-generated name '%s'", name
            )

        existing = [c.name for c in self._client.get_collections().collections]
        if name in existing:
            logger.info("Using existing collection '%s'", name)
        else:
            self._client.create_collection(
                collection_name=name,
                vectors_config=self._models.VectorParams(
                    size=VECTOR_DIM,
                    distance=self._models.Distance.COSINE,
                ),
            )
            logger.info("Created fresh collection '%s' (dim=%d)", name, VECTOR_DIM)
        return name

    def search(self, query_text: str, top_k: int = 10, user_id: str = "") -> list[dict]:
        vector = _embed([query_text], self._config)[0]
        query_filter = None
        if user_id:
            query_filter = self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="user_id",
                        match=self._models.MatchValue(value=user_id),
                    )
                ]
            )
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=0.3,
        )
        return [
            {
                "id": str(r.id),
                "score": round(r.score, 4),
                "memory": r.payload.get("content", ""),
                "category": r.payload.get("category", ""),
                "created_at": r.payload.get("created_at", ""),
            }
            for r in results.points
        ]

    def get_all(self, user_id: str = "", limit: int = 100) -> list[dict]:
        query_filter = None
        if user_id:
            query_filter = self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="user_id",
                        match=self._models.MatchValue(value=user_id),
                    )
                ]
            )
        results, _ = self._client.scroll(
            collection_name=self._collection,
            limit=limit,
            scroll_filter=query_filter,
            with_payload=True,
        )
        return [
            {
                "id": str(r.id),
                "memory": r.payload.get("content", ""),
                "category": r.payload.get("category", ""),
                "created_at": r.payload.get("created_at", ""),
            }
            for r in results
        ]

    def add(self, content: str, user_id: str, agent_id: str, category: str = "fact") -> str:
        vector = _embed([content], self._config)[0]
        point_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._client.upsert(
            collection_name=self._collection,
            points=[
                self._models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": content,
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "category": category,
                        "created_at": now,
                    },
                )
            ],
        )
        return point_id

    def get_point(self, point_id: str) -> Optional[dict]:
        """Retrieve a single point's payload by ID. Returns None if not found."""
        try:
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id],
                with_payload=True,
            )
            if not points:
                return None
            p = points[0]
            if p.payload is None:
                return None
            return {
                "id": str(p.id),
                "content": p.payload.get("content", ""),
                "category": p.payload.get("category", ""),
                "created_at": p.payload.get("created_at", ""),
            }
        except Exception:
            return None

    def delete(self, point_id: str) -> bool:
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=self._models.PointIdsList(
                    points=[point_id],
                ),
            )
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class QdrantMemoryProvider(MemoryProvider):
    """Qdrant vector memory — local Qdrant + any OpenAI-compatible embedding API."""

    def __init__(self):
        self._config = None
        self._store: Optional[_QdrantStore] = None
        self._indexer: Optional[FileIndexer] = None
        self._learning_store: Optional[LearningStore] = None
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
        cfg = _load_config()
        return bool(cfg.get("embedding_base_url")) and bool(cfg.get("embedding_api_key"))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._session_id = session_id
        self._store = _QdrantStore(self._config)
        self._indexer = FileIndexer(
            self._store,
            embed_fn=lambda texts: _embed(texts, self._config),
            config=self._config,
        )
        self._learning_store = LearningStore(
            self._store,
            embed_fn=lambda texts: _embed(texts, self._config),
            config=self._config,
        )
        try:
            self._learning_store.ensure_collection()
        except Exception:
            logger.debug("Learning collection setup failed — learning tools unavailable", exc_info=True)
        self._consolidation = ConsolidationEngine(
            self._store,
            embed_fn=lambda texts: _embed(texts, self._config),
            learning_store=self._learning_store,
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
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Qdrant circuit breaker tripped after %d consecutive failures. "
                "Pausing for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    # ── System prompt ─────────────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        coll = self._store._collection if self._store else "?"
        return (
            "# Qdrant Memory\n"
            f"Active. Qdrant collection: {coll} (dim={VECTOR_DIM}).\n"
            "Use qdrant_search to find memories, qdrant_remember to store facts, "
            "qdrant_profile for a full overview, qdrant_forget to delete."
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

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="qdrant-prefetch")
        self._prefetch_thread.start()

    # ── Turn sync ─────────────────────────────────────────────────────────

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Extract and store facts from the conversation turn."""
        if self._is_breaker_open() or not self._store:
            return

        def _sync():
            try:
                # Simple extraction — store notable user statements
                # Full LLM-based extraction could be added later
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
            LEARNING_STORE_SCHEMA, LEARNING_SEARCH_SCHEMA,
            LEARNING_PREVIEW_SCHEMA, LEARNING_APPROVE_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Qdrant memory temporarily unavailable (multiple consecutive failures)."
            })

        if not self._store:
            return tool_error("Qdrant store not initialized")

        try:
            if tool_name == "qdrant_profile":
                memories = self._store.get_all(user_id=self._user_id)
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m["memory"] for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})

            elif tool_name == "qdrant_search":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                top_k = min(int(args.get("top_k", 10)), 50)
                results = self._store.search(query, top_k=top_k, user_id=self._user_id)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [
                    {
                        "memory": r["memory"],
                        "score": r["score"],
                        "category": r.get("category", ""),
                    }
                    for r in results
                ]
                return json.dumps({"results": items, "count": len(items)})

            elif tool_name == "qdrant_remember":
                content = args.get("content", "")
                if not content:
                    return tool_error("Missing required parameter: content")
                category = args.get("category", "fact")
                point_id = self._store.add(
                    content=content,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    category=category,
                )
                self._record_success()
                return json.dumps({"result": "Memory stored.", "id": point_id})

            elif tool_name == "qdrant_forget":
                point_id = args.get("point_id", "")
                if not point_id:
                    return tool_error("Missing required parameter: point_id")
                dry_run = args.get("dry_run", True)  # Default: safe

                if dry_run:
                    # Preview what would be deleted
                    memory = self._store.get_point(point_id)
                    if memory:
                        return json.dumps({
                            "dry_run": True,
                            "result": f"Would delete: [{memory.get('category', '?')}] {memory.get('content', '?')[:120]}",
                            "point_id": point_id,
                            "category": memory.get("category", ""),
                            "created_at": memory.get("created_at", ""),
                        })
                    return json.dumps({"dry_run": True, "result": "Memory not found. Nothing to delete."})

                # Live deletion — user explicitly opted in
                ok = self._store.delete(point_id)
                self._record_success()
                if ok:
                    return json.dumps({"result": "Memory deleted.", "id": point_id})
                return json.dumps({"result": "Memory not found or already deleted."})

            elif tool_name == "qdrant_index":
                if not self._indexer:
                    return tool_error("Indexer not initialized")
                paths = args.get("paths", [])
                if not paths:
                    return tool_error("Missing required parameter: paths")
                dry_run = args.get("dry_run", True)
                max_files = args.get("max_files")

                result = self._indexer.index(
                    paths=paths,
                    dry_run=dry_run,
                    max_files=max_files,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                )
                self._record_success()
                return json.dumps(result)

            # ── Consolidation ──────────────────────────────────────────

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

            # ── Learning tools ──────────────────────────────────────────

            elif tool_name == "qdrant_learning_store":
                if not self._learning_store:
                    return tool_error("Learning store not initialized")
                result = self._learning_store.store_learning(
                    lesson=args["lesson"],
                    learning_type=args.get("learning_type", "workflow_lesson"),
                    trigger=args.get("trigger", ""),
                    mistake=args.get("mistake", ""),
                    correction=args.get("correction", ""),
                    evidence=args.get("evidence", ""),
                    tool_name=args.get("tool_name", ""),
                    command=args.get("command", ""),
                    importance=int(args.get("importance", 7)),
                    confidence=float(args.get("confidence", 0.8)),
                    tags=args.get("tags", []),
                    promote_to_skill_candidate=args.get("promote_to_skill", False),
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                )
                self._record_success()
                return json.dumps({"result": "Learning stored.", **result})

            elif tool_name == "qdrant_learning_search":
                if not self._learning_store:
                    return tool_error("Learning store not initialized")
                results = self._learning_store.search(
                    query=args["query"],
                    top_k=int(args.get("top_k", 5)),
                    learning_type=args.get("learning_type", ""),
                    user_id=self._user_id,
                )
                self._record_success()
                return json.dumps({"results": results, "count": len(results)})

            elif tool_name == "qdrant_learning_preview":
                if not self._learning_store:
                    return tool_error("Learning store not initialized")
                pending = self._learning_store.get_pending()
                return json.dumps({"pending": pending, "count": len(pending)})

            elif tool_name == "qdrant_learning_approve":
                if not self._learning_store:
                    return tool_error("Learning store not initialized")
                cid = args["candidate_id"]
                dry_run = args.get("dry_run", True)
                if dry_run:
                    pending = self._learning_store.get_pending()
                    target = next((c for c in pending if c.get("candidate_id") == cid), None)
                    if target:
                        return json.dumps({
                            "dry_run": True,
                            "result": f"Would approve: {target.get('lesson', '?')[:120]}",
                            "candidate": target,
                        })
                    return json.dumps({"dry_run": True, "result": "Candidate not found."})

                result = self._learning_store.approve_candidate(
                    cid,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                )
                if result:
                    self._record_success()
                    return json.dumps({"result": "Candidate approved and stored.", **result})
                return json.dumps({"result": "Candidate not found."})

            return tool_error(f"Unknown tool: {tool_name}")

        except Exception as e:
            self._record_failure()
            return tool_error(f"Qdrant memory error: {e}")

    # ── Config ────────────────────────────────────────────────────────────

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "qdrant_url",
                "description": "Qdrant server URL",
                "default": "http://localhost:6333",
            },
            {
                "key": "qdrant_api_key",
                "description": "Qdrant API key (optional)",
                "secret": True,
                "env_var": "QDRANT_API_KEY",
            },
            {
                "key": "embedding_base_url",
                "description": "OpenAI-compatible embedding endpoint URL",
                "required": True,
                "env_var": "EMBEDDING_BASE_URL",
            },
            {
                "key": "embedding_api_key",
                "description": "API key for the embedding service",
                "secret": True,
                "required": True,
                "env_var": "EMBEDDING_API_KEY",
            },
            {
                "key": "embedding_model",
                "description": "Embedding model name",
                "default": "doubao-embedding-vision",
            },
            {
                "key": "collection_name",
                "description": "Unique collection name (auto-generated if empty). MUST be unique per deployment.",
                "env_var": "QDRANT_COLLECTION",
            },
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

        # Separate secrets
        secrets_map = {"qdrant_api_key": "QDRANT_API_KEY", "embedding_api_key": "EMBEDDING_API_KEY"}
        secrets_to_env = {}
        non_secrets = dict(existing)

        for key, env_var in secrets_map.items():
            if key in values and values[key]:
                secrets_to_env[env_var] = values.pop(key, "")
                non_secrets.pop(key, None)

        # Write non-secrets to JSON
        non_secrets.update({k: v for k, v in values.items() if v is not None and v != ""})
        if non_secrets:
            config_path.write_text(json.dumps(non_secrets, indent=2))
        elif config_path.exists():
            config_path.unlink()

        # Write secrets to .env
        if secrets_to_env:
            from pathlib import Path as P
            env_path = P(hermes_home) / ".env"
            env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
            for k, v in secrets_to_env.items():
                if v:
                    line = f"{k}={v}"
                    import re
                    if re.search(rf"^{re.escape(k)}=", env_text, re.MULTILINE):
                        env_text = re.sub(
                            rf"^{re.escape(k)}=.*",
                            line,
                            env_text,
                            flags=re.MULTILINE,
                        )
                    else:
                        env_text += f"\n{line}"
            env_path.write_text(env_text.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the Qdrant memory provider plugin."""
    ctx.register_memory_provider(QdrantMemoryProvider())

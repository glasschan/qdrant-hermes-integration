"""Learning collection for Qdrant memory plugin.

Separate collection (`<main_collection>_learnings`) for procedural
learnings — lessons from tool failures, user corrections, workflows,
and environment quirks. Manual/gated by default; auto-extraction
disabled until explicitly enabled.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from qdrant_client.http import models

logger = logging.getLogger(__name__)

LEARNING_TYPES = [
    "tool_failure_lesson",
    "user_correction",
    "workflow_lesson",
    "environment_quirk",
]


class LearningStore:
    """Structured procedural learning storage in a dedicated Qdrant collection."""

    def __init__(
        self,
        store,          # _QdrantStore instance
        embed_fn,       # embedding function
        config: dict,
    ):
        self._store = store
        self._embed = embed_fn
        self._config = config
        self._main_collection = store._collection
        self._collection = f"{self._main_collection}_learnings"
        self._vector_dim = int(config.get("vector_size", 2048))
        self._pending_candidates: dict[str, dict] = {}  # {candidate_id: candidate}

    @property
    def collection_name(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """Create the learning collection if it doesn't exist."""
        existing = [c.name for c in self._store._client.get_collections().collections]
        if self._collection not in existing:
            self._store._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._vector_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created learning collection '%s'", self._collection)

    # ── store ─────────────────────────────────────────────────────────────

    def store_learning(
        self,
        lesson: str,
        learning_type: str = "workflow_lesson",
        trigger: str = "",
        mistake: str = "",
        correction: str = "",
        evidence: str = "",
        tool_name: str = "",
        command: str = "",
        importance: int = 7,
        confidence: float = 0.8,
        tags: Optional[list[str]] = None,
        promote_to_skill_candidate: bool = False,
        user_id: str = "",
        agent_id: str = "",
    ) -> dict:
        """Store a procedural learning. Returns the created point info."""
        if learning_type not in LEARNING_TYPES:
            learning_type = "workflow_lesson"  # fallback

        # Build searchable text from structured fields
        searchable = _build_searchable_text(
            lesson=lesson,
            learning_type=learning_type,
            trigger=trigger,
            mistake=mistake,
            correction=correction,
            tool_name=tool_name,
        )

        vector = self._embed([searchable])[0]
        point_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self._store._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": searchable,
                        "lesson": lesson,
                        "learning_type": learning_type,
                        "trigger": trigger,
                        "mistake": mistake,
                        "correction": correction,
                        "evidence": evidence,
                        "tool_name": tool_name,
                        "command": command,
                        "importance": importance,
                        "confidence": confidence,
                        "tags": tags or [],
                        "promote_to_skill": promote_to_skill_candidate,
                        "source_type": "learning",
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "created_at": now,
                    },
                )
            ],
        )

        return {
            "id": point_id,
            "lesson": lesson[:120],
            "learning_type": learning_type,
            "collection": self._collection,
        }

    # ── search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        learning_type: str = "",
        user_id: str = "",
    ) -> list[dict]:
        """Semantic search over procedural learnings."""
        vector = self._embed([query])[0]

        query_filter = None
        conditions = []
        if learning_type:
            conditions.append(
                models.FieldCondition(
                    key="learning_type",
                    match=models.MatchValue(value=learning_type),
                )
            )
        if user_id:
            conditions.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            )
        if conditions:
            query_filter = models.Filter(must=conditions)

        results = self._store._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=min(top_k, 20),
            query_filter=query_filter,
            with_payload=True,
            score_threshold=0.3,
        )

        return [
            {
                "id": str(r.id),
                "score": round(r.score, 4),
                "lesson": r.payload.get("lesson", ""),
                "learning_type": r.payload.get("learning_type", ""),
                "mistake": r.payload.get("mistake", ""),
                "correction": r.payload.get("correction", ""),
                "trigger": r.payload.get("trigger", ""),
                "importance": r.payload.get("importance", 7),
                "created_at": r.payload.get("created_at", ""),
            }
            for r in results.points
        ]

    # ── pending candidates (gated auto-extraction) ─────────────────────────

    def add_candidate(self, candidate: dict) -> str:
        """Add a pending learning candidate for review. Returns candidate_id."""
        cid = str(uuid.uuid4())[:8]
        candidate["candidate_id"] = cid
        candidate["created_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_candidates[cid] = candidate
        return cid

    def get_pending(self) -> list[dict]:
        """Get all pending candidates (preview, never writes)."""
        return list(self._pending_candidates.values())

    def approve_candidate(
        self,
        candidate_id: str,
        user_id: str = "",
        agent_id: str = "",
    ) -> Optional[dict]:
        """Approve a pending candidate and store it. Returns stored info or None."""
        candidate = self._pending_candidates.pop(candidate_id, None)
        if not candidate:
            return None

        return self.store_learning(
            lesson=candidate.get("lesson", ""),
            learning_type=candidate.get("learning_type", "workflow_lesson"),
            trigger=candidate.get("trigger", ""),
            mistake=candidate.get("mistake", ""),
            correction=candidate.get("correction", ""),
            evidence=candidate.get("evidence", ""),
            tool_name=candidate.get("tool_name", ""),
            command=candidate.get("command", ""),
            importance=candidate.get("importance", 7),
            confidence=candidate.get("confidence", 0.8),
            tags=candidate.get("tags", []),
            promote_to_skill_candidate=candidate.get("promote_to_skill_candidate", False),
            user_id=user_id,
            agent_id=agent_id,
        )

    def reject_candidate(self, candidate_id: str) -> bool:
        """Reject/discard a pending candidate. Returns True if found."""
        return self._pending_candidates.pop(candidate_id, None) is not None


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_searchable_text(
    lesson: str,
    learning_type: str,
    trigger: str = "",
    mistake: str = "",
    correction: str = "",
    tool_name: str = "",
) -> str:
    """Build embedding text from structured learning fields."""
    parts = [f"[{learning_type}] {lesson}"]
    if trigger:
        parts.append(f"Trigger: {trigger}")
    if mistake:
        parts.append(f"Mistake: {mistake}")
    if correction:
        parts.append(f"Correction: {correction}")
    if tool_name:
        parts.append(f"Tool: {tool_name}")
    return "\n".join(parts)

"""Qdrant client wrapper — thin layer around qdrant-client.

All operations scoped to a single collection (self._collection).
NEVER touches another collection. NEVER deletes collections.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from datetime import datetime, timezone
from typing import Optional

from plugin.config import VECTOR_DIM
from plugin.embeddings import embed

logger = logging.getLogger(__name__)


class QdrantStore:
    """Thin wrapper around qdrant-client for a single collection."""

    def __init__(self, config: dict):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._models = models
        url = config["qdrant_url"]
        api_key = config.get("qdrant_api_key") or None
        self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._config = config
        self._collection = self._ensure_collection()

    @property
    def collection(self) -> str:
        return self._collection

    def _ensure_collection(self) -> str:
        """Use/create OUR collection. NEVER delete any collection."""
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

    # ── CRUD ────────────────────────────────────────────────────────────

    def search(self, query_text: str, top_k: int = 10, user_id: str = "") -> list[dict]:
        vector = embed([query_text], self._config)[0]
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
        vector = embed([content], self._config)[0]
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
                points_selector=self._models.PointIdsList(points=[point_id]),
            )
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

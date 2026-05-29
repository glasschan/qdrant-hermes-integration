"""Qdrant client wrapper — thin layer around qdrant-client.

All operations scoped to a single collection (self._collection).
NEVER touches another collection. NEVER deletes collections.

Memory hygiene features:
- Pre-save dedup: add() searches for semantically similar points before creating new
- Payload metadata: created_at + updated_at + version for every point
- Payload indexes: category + updated_at for fast filtered search
- Recency-weighted search: optional time decay in ranking
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional

# localhost Qdrant + API key triggers a benign warning about insecure
# connection. The traffic never leaves the machine — suppress locally.
import warnings

from .config import VECTOR_DIM, DEDUP_THRESHOLD, DEDUP_ENABLED
from .embeddings import embed

logger = logging.getLogger(__name__)


class QdrantStore:
    """Thin wrapper around qdrant-client for a single collection."""

    def __init__(self, config: dict):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._models = models
        url = config["qdrant_url"]
        api_key = config.get("qdrant_api_key") or None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", message="Api key is used with an insecure connection")
            self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._config = config
        self._collection = self._ensure_collection()
        self._ensure_payload_indexes()

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def client(self):
        """Expose the Qdrant client for sub-components (indexer, consolidation)."""
        return self._client

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

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for fields we filter/search by."""
        try:
            # Get existing indexes
            existing = {f.field_name for f in self._client.get_collection(
                collection_name=self._collection
            ).config.params.get("payload_schema", {})}

            indexes_to_create = {
                "category": "keyword",
                "updated_at": "integer",
                "priority": "integer",
                "origin": "keyword",
            }

            for field_name, field_type in indexes_to_create.items():
                if field_name not in existing:
                    try:
                        self._client.create_payload_index(
                            collection_name=self._collection,
                            field_name=field_name,
                            field_type=field_type,
                        )
                        logger.info("Created payload index on '%s'", field_name)
                    except Exception:
                        logger.debug("Payload index '%s' may already exist", field_name, exc_info=True)

        except Exception:
            logger.debug("Payload index creation skipped (may already exist)", exc_info=True)

    # ── CRUD ────────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        user_id: str = "",
        recency_weight: float | None = None,
        tags: list[str] | None = None,
        category_boost: dict | None = None,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """Search memory with optional recency weighting, tag filtering, and category boost.

        recency_weight: 0.0 = pure relevance, 1.0 = 50/50 relevance + freshness.
        If None, falls back to config value.
        tags: Optional list of tags to filter by (AND logic — all must match).
        category_boost: dict mapping category name → score multiplier (e.g. {"correction": 1.3}).
        score_threshold: Minimum cosine score to include.
        """
        vector = embed([query_text], self._config)[0]

        must_conditions = []
        if user_id:
            must_conditions.append(
                self._models.FieldCondition(
                    key="user_id",
                    match=self._models.MatchValue(value=user_id),
                )
            )
        if tags:
            for tag in tags:
                must_conditions.append(
                    self._models.FieldCondition(
                        key="tags",
                        match=self._models.MatchValue(value=tag),
                    )
                )

        query_filter = None
        if must_conditions:
            query_filter = self._models.Filter(must=must_conditions)
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k * 2,  # fetch extra for recency + boost re-rank
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        if recency_weight is None:
            recency_weight = float(self._config.get("search_recency_weight", 0.0))

        # Merge config-level boost with caller-provided boost
        effective_boost = dict(self._config.get("prefetch_category_boost", {}))
        if category_boost:
            effective_boost.update(category_boost)

        entries = []
        now = datetime.now(timezone.utc)
        for r in results.points:
            score = round(r.score, 4)
            category = r.payload.get("category", "")

            # Apply category boost (corrections/instructions surface higher)
            if effective_boost and category in effective_boost:
                score = round(score * effective_boost[category], 4)

            # Priority boost: higher priority (lower number) → slight score bump
            priority = int(r.payload.get("priority", 3))
            if priority <= 2:
                score = round(score * (1.0 + (3 - priority) * 0.05), 4)

            created_raw = r.payload.get("updated_at") or r.payload.get("created_at", "")
            # Apply recency decay if weight > 0
            if recency_weight > 0 and created_raw:
                try:
                    created_dt = datetime.fromisoformat(created_raw)
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    age_days = (now - created_dt).total_seconds() / 86400
                    # Freshness score: 1.0 for today, decays to ~0.0 after 90 days
                    freshness = max(0.0, 1.0 - age_days / 90.0)
                    # Blend: recency_weight controls how much freshness matters
                    score = score * (1 - recency_weight) + freshness * recency_weight
                except (ValueError, TypeError):
                    pass

            entries.append(
                {
                    "id": str(r.id),
                    "score": round(score, 4),
                    "memory": r.payload.get("content", ""),
                    "category": r.payload.get("category", ""),
                    "tags": r.payload.get("tags", []),
                    "version": r.payload.get("version", 1),
                    "created_at": r.payload.get("created_at", ""),
                    "updated_at": r.payload.get("updated_at", ""),
                }
            )

        # Sort by blended score, take top_k
        entries.sort(key=lambda x: x["score"], reverse=True)
        return entries[:top_k]

    def get_all(self, user_id: str = "", limit: int = 0) -> list[dict]:
        """Retrieve all memories with scroll-based pagination.

        Args:
            user_id: Optional user filter.
            limit: Max results. 0 = no limit (fetch all). Any positive value caps results.
        """
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

        all_results = []
        offset = None
        page_size = min(limit, 500) if limit > 0 else 500

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=page_size,
                offset=offset,
                scroll_filter=query_filter,
                with_payload=True,
            )
            all_results.extend(results)

            if not next_offset or not results:
                break
            if limit > 0 and len(all_results) >= limit:
                all_results = all_results[:limit]
                break
            offset = next_offset

        return [
            {
                "id": str(r.id),
                "memory": r.payload.get("content", ""),
                "category": r.payload.get("category", ""),
                "tags": r.payload.get("tags", []),
                "version": r.payload.get("version", 1),
                "created_at": r.payload.get("created_at", ""),
                "updated_at": r.payload.get("updated_at", ""),
            }
            for r in all_results
        ]

    def add(
        self,
        content: str,
        user_id: str,
        agent_id: str,
        category: str = "fact",
        tags: list[str] | None = None,
        priority: int = 3,
        origin: str = "explicit",
    ) -> str:
        """Store a memory with pre-save dedup.

        If dedup_enabled and a semantically similar point exists (>threshold),
        updates the existing point's payload instead of creating a new one.
        Returns the point ID (existing or new).
        """
        vector = embed([content], self._config)[0]
        now = datetime.now(timezone.utc).isoformat()

        # Pre-save dedup: search for existing similar content
        dedup_enabled = self._config.get("dedup_enabled", DEDUP_ENABLED)
        dedup_threshold = float(self._config.get("dedup_threshold", DEDUP_THRESHOLD))

        if dedup_enabled:
            existing = self._find_duplicate(
                vector=vector,
                user_id=user_id,
                threshold=dedup_threshold,
            )
            if existing:
                # Update existing point — upsert with NEW vector + payload
                # (set_payload alone would leave the vector stale, breaking search accuracy)
                point_id = existing["id"]
                curr_version = existing.get("version", 1)
                self._client.upsert(
                    collection_name=self._collection,
                    points=[
                        self._models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "content": content,
                                "category": category,
                                "tags": tags or existing.get("tags", []),
                                "user_id": user_id,
                                "agent_id": agent_id,
                                "priority": priority,
                                "origin": origin,
                                "created_at": existing.get("created_at", now),
                                "updated_at": now,
                                "version": curr_version + 1,
                            },
                        )
                    ],
                )
                logger.debug(
                    "Dedup: upserted existing point %s with new vector (v%d → v%d)",
                    point_id, curr_version, curr_version + 1,
                )
                return point_id

        # No duplicate found — create new point
        point_id = str(uuid.uuid4())
        payload = {
            "content": content,
            "user_id": user_id,
            "agent_id": agent_id,
            "category": category,
            "tags": tags or [],
            "priority": priority,
            "origin": origin,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
        self._client.upsert(
            collection_name=self._collection,
            points=[
                self._models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return point_id

    def _find_duplicate(
        self,
        vector: list[float],
        user_id: str,
        threshold: float,
    ) -> dict | None:
        """Search for an existing point with similar vector.

        Returns the best-matching existing point dict if above threshold.
        Uses limit=3 and picks the highest-scoring match.
        """
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
            limit=3,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=threshold,
        )
        if results.points:
            best = results.points[0]
            return {
                "id": str(best.id),
                "score": round(best.score, 4),
                "version": best.payload.get("version", 1),
                "tags": best.payload.get("tags", []),
                "created_at": best.payload.get("created_at", ""),
                "updated_at": best.payload.get("updated_at", ""),
            }
        return None

    def update_payload(self, point_id: str, payload: dict) -> bool:
        """Update payload fields on an existing point without re-embedding.

        Uses Qdrant's set_payload to add/overwrite fields atomically.
        """
        try:
            self._client.set_payload(
                collection_name=self._collection,
                points=[point_id],
                payload=payload,
            )
            return True
        except Exception as e:
            logger.warning("update_payload failed for %s: %s", point_id, e)
            return False

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
                "tags": p.payload.get("tags", []),
                "version": p.payload.get("version", 1),
                "created_at": p.payload.get("created_at", ""),
                "updated_at": p.payload.get("updated_at", ""),
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
        except Exception as e:
            logger.warning("delete failed for %s: %s", point_id, e)
            return False

    def close(self) -> None:
        self._client.close()
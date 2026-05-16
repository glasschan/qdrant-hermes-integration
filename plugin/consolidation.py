"""Consolidation for Qdrant memory plugin.

Report-only memory consolidation — finds duplicates, stale memories,
and potential quality issues. NEVER mutates data automatically.
All reports are read-only by design.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from qdrant_client.http import models

logger = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────

DUPLICATE_THRESHOLD = 0.92       # cosine similarity threshold for duplicates
STALE_DAYS = 90                  # memories older than this are "stale"
MIN_IMPORTANCE_FOR_KEEP = 4      # below this + stale = low-value candidate
MAX_POINTS_TO_SCAN = 500
MAX_DUPLICATE_GROUPS = 20


# ── ConsolidationEngine ──────────────────────────────────────────────────────

class ConsolidationEngine:
    """Report-only consolidation. Finds issues, never mutates."""

    def __init__(
        self,
        store,          # _QdrantStore instance
        embed_fn,       # embedding function
        learning_store=None,  # optional LearningStore
    ):
        self._store = store
        self._embed = embed_fn
        self._learning = learning_store
        self._collection = store.collection

    def consolidate(
        self,
        scope: str = "memory",          # "memory" | "learning" | "both"
        max_points: int = MAX_POINTS_TO_SCAN,
        max_groups: int = MAX_DUPLICATE_GROUPS,
        include_examples: bool = False,
    ) -> dict:
        """Generate a consolidation report. Read-only, never mutates.

        Returns:
            report_id: str
            generated_at: str (ISO timestamp)
            scope: str
            points_scanned: int
            proposals: list[dict] — each a suggested action (duplicate, stale, quality)
            total_proposals: int
        """
        report_id = _generate_report_id()
        now = datetime.now(timezone.utc).isoformat()

        proposals = []

        if scope in ("memory", "both"):
            # Scan main memory collection
            mem_proposals = self._scan_collection(
                self._collection,
                max_points=max_points,
                max_groups=max_groups,
                include_examples=include_examples,
            )
            proposals.extend(mem_proposals)

        if scope in ("learning", "both") and self._learning:
            # Scan learning collection
            learn_proposals = self._scan_collection(
                self._learning.collection_name,
                max_points=max_points,
                max_groups=max_groups,
                include_examples=include_examples,
            )
            proposals.extend(learn_proposals)

        return {
            "report_id": report_id,
            "generated_at": now,
            "scope": scope,
            "points_scanned": sum(p.get("points_considered", 0) for p in proposals) or max_points,
            "proposals": proposals,
            "total_proposals": len(proposals),
        }

    def _scan_collection(
        self,
        collection_name: str,
        max_points: int,
        max_groups: int,
        include_examples: bool = False,
    ) -> list[dict]:
        """Scan a collection for duplicates and stale memories."""
        proposals = []

        try:
            points = self._fetch_all_points(collection_name, max_points)
            if not points:
                return proposals

            # 1. Duplicate detection
            dup_groups = self._find_duplicates(points, max_groups, include_examples)

            # 2. Stale detection
            stale = self._find_stale(points, include_examples)

            # 3. Quality warnings
            quality = self._find_quality_issues(points, include_examples)

            if dup_groups:
                proposals.append({
                    "type": "duplicate_clusters",
                    "collection": collection_name,
                    "groups": dup_groups,
                    "count": len(dup_groups),
                    "points_considered": len(points),
                })

            if stale:
                proposals.append({
                    "type": "stale_low_value",
                    "collection": collection_name,
                    "candidates": stale,
                    "count": len(stale),
                    "points_considered": len(points),
                })

            if quality:
                proposals.append({
                    "type": "quality_warnings",
                    "collection": collection_name,
                    "warnings": quality,
                    "count": len(quality),
                    "points_considered": len(points),
                })

        except Exception as e:
            logger.warning("Consolidation scan failed for %s: %s", collection_name, e)

        return proposals

    # ── internals ─────────────────────────────────────────────────────────

    def _fetch_all_points(self, collection_name: str, limit: int) -> list[dict]:
        """Scroll all points from a collection."""
        try:
            results, _ = self._store._client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            points = []
            for r in results:
                p = {
                    "id": str(r.id),
                    "content": r.payload.get("content", "") or "",
                    "category": r.payload.get("category", ""),
                    "created_at": r.payload.get("created_at", ""),
                    "importance": int(r.payload.get("importance", 5)),
                    "source_type": r.payload.get("source_type", ""),
                    "vector": r.vector,
                }
                points.append(p)
            return points
        except Exception:
            return []

    def _find_duplicates(
        self, points: list[dict], max_groups: int, include_examples: bool
    ) -> list[dict]:
        """Find groups of semantically similar points."""
        if len(points) < 2:
            return []

        # Compare pairs using vector cosine similarity
        groups = []
        seen = set()

        for i in range(len(points)):
            if points[i]["id"] in seen:
                continue
            group = [points[i]]
            vi = points[i].get("vector")
            if not vi:
                continue

            for j in range(i + 1, len(points)):
                if points[j]["id"] in seen:
                    continue
                vj = points[j].get("vector")
                if not vj:
                    continue

                sim = _cosine_similarity(vi, vj)
                if sim >= DUPLICATE_THRESHOLD:
                    group.append(points[j])
                    seen.add(points[j]["id"])

            if len(group) >= 2:
                seen.add(points[i]["id"])
                g = {
                    "size": len(group),
                    "avg_similarity": round(_avg_group_similarity(group), 4),
                    "members": [
                        {
                            "id": p["id"],
                            "content_preview": p["content"][:80],
                            "created_at": p["created_at"],
                        }
                        for p in group[:5]
                    ],
                }
                if include_examples:
                    g["examples"] = [p["content"][:200] for p in group[:3]]
                groups.append(g)

            if len(groups) >= max_groups:
                break

        return groups

    def _find_stale(self, points: list[dict], include_examples: bool) -> list[dict]:
        """Find old, low-importance memories."""
        stale = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)

        for p in points:
            created = p.get("created_at", "")
            try:
                created_dt = datetime.fromisoformat(created)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            importance = int(p.get("importance", 5))
            if created_dt < cutoff and importance < MIN_IMPORTANCE_FOR_KEEP:
                stale.append({
                    "id": p["id"],
                    "content_preview": p["content"][:80],
                    "category": p.get("category", ""),
                    "importance": importance,
                    "created_at": p["created_at"],
                    "age_days": (datetime.now(timezone.utc) - created_dt).days,
                })

        return sorted(stale, key=lambda x: x["age_days"], reverse=True)[:50]

    def _find_quality_issues(
        self, points: list[dict], include_examples: bool
    ) -> list[dict]:
        """Find potential quality issues (very short content, possible secrets)."""
        warnings = []

        SECRET_PATTERNS = ["sk-", "api_key", "token", "password", "secret", "BEGIN RSA", "BEGIN OPENSSH"]
        for p in points:
            content = p.get("content", "")
            issues = []

            # Very short / low-quality content
            if len(content) < 20 and p.get("source_type") != "file":
                issues.append("very_short_content")

            # Possible secrets
            lower = content.lower()
            for pattern in SECRET_PATTERNS:
                if pattern.lower() in lower:
                    issues.append(f"possible_secret:{pattern}")
                    break

            if issues:
                w = {
                    "id": p["id"],
                    "content_preview": content[:80],
                    "issues": issues,
                    "created_at": p.get("created_at", ""),
                }
                if include_examples:
                    w["content_redacted"] = _redact_secrets(content)[:200]
                warnings.append(w)

        return warnings[:20]


# ── math helpers ─────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _avg_group_similarity(group: list[dict]) -> float:
    """Average pairwise cosine similarity within a group."""
    if len(group) < 2:
        return 0.0
    sims = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            vi = group[i].get("vector")
            vj = group[j].get("vector")
            if vi and vj:
                sims.append(_cosine_similarity(vi, vj))
    return sum(sims) / len(sims) if sims else 0.0


def _generate_report_id() -> str:
    """Generate a unique report ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(ts.encode()).hexdigest()[:6]
    return f"report-{ts}-{suffix}"


def _redact_secrets(text: str) -> str:
    """Redact common secret patterns from text for safe preview."""
    import re
    text = re.sub(r'(sk-[a-zA-Z0-9]{10,})', '[REDACTED_KEY]', text)
    text = re.sub(r'(api_key[=:]\s*["\']?)([^"\'\s]+)', r'\1[REDACTED]', text)
    text = re.sub(r'-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----',
                   '[REDACTED_PRIVATE_KEY]', text, flags=re.DOTALL)
    return text

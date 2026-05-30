"""Consolidation for Qdrant memory plugin.

Report-only memory consolidation — finds duplicates, stale memories,
and potential quality issues. NEVER mutates data automatically.

v0.7.0: Quick mode (skip expensive dedup), evolved_from tracking.
v0.8.0: Topic clustering integration, dedup quality score.
v0.9.0: Numpy cosine, incremental consolidation, auto-stale/auto-prune lifecycle.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Optional


from .config import (
    CONSOLIDATION_METADATA_ID,
    AUTO_STALE_DAYS,
    AUTO_PRUNE_DAYS,
)

logger = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────

DUPLICATE_THRESHOLD = 0.92       # cosine similarity threshold for duplicates
STALE_DAYS = 90                  # memories older than this are "stale"
MIN_IMPORTANCE_FOR_KEEP = 4      # below this + stale = low-value candidate (legacy)
MIN_PRIORITY_FOR_STALE = 4       # priority >= this + stale = low-value candidate
MAX_POINTS_TO_SCAN = 500
MAX_DUPLICATE_GROUPS = 20
STALE_CATEGORIES_TO_SKIP = ["conversation"]  # always flag these as stale


# ── Numpy cosine similarity (v0.9.0) ─────────────────────────────────────────

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses numpy when available for batch performance, falls back to pure Python.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    if _HAS_NUMPY:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        dot = np.dot(va, vb)
        mag_a = np.linalg.norm(va)
        mag_b = np.linalg.norm(vb)
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return float(dot / (mag_a * mag_b))
    else:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


def _batch_cosine_similarity(vectors: list[list[float]]) -> list[list[float]]:
    """Compute pairwise cosine similarity for all vectors at once (v0.9.0).

    Returns a symmetric matrix of similarities.
    Uses numpy for O(n²) batch computation when available.
    """
    if not vectors:
        return []

    if _HAS_NUMPY and len(vectors) > 1:
        mat = np.asarray(vectors, dtype=np.float64)
        # Normalize rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = mat / norms
        # Pairwise cosine = normed @ normed.T
        sim_matrix = np.dot(normed, normed.T)
        return sim_matrix.tolist()
    else:
        # Pure Python fallback
        n = len(vectors)
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            result[i][i] = 1.0
            for j in range(i + 1, n):
                s = _cosine_similarity(vectors[i], vectors[j])
                result[i][j] = s
                result[j][i] = s
        return result


def _avg_group_similarity(group: list[dict]) -> float:
    """Average pairwise cosine similarity within a group.

    Uses numpy batch computation when available.
    """
    if len(group) < 2:
        return 0.0

    vecs = [p.get("vector") for p in group if p.get("vector")]
    if len(vecs) < 2:
        return 0.0

    if _HAS_NUMPY:
        mat = np.asarray(vecs, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = mat / norms
        sim_matrix = np.dot(normed, normed.T)
        n = len(vecs)
        # Sum upper triangle (excluding diagonal) and divide by pair count
        total = float(np.sum(np.triu(sim_matrix, k=1)))
        count = n * (n - 1) / 2
        return total / count if count > 0 else 0.0
    else:
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(_cosine_similarity(vecs[i], vecs[j]))
        return sum(sims) / len(sims) if sims else 0.0


# ── ConsolidationEngine ──────────────────────────────────────────────────────

class ConsolidationEngine:
    """Report-only consolidation. Finds issues, never mutates by default.
    When auto_stale/auto_prune are enabled, applies lifecycle mutations.

    v0.9.0: Supports incremental consolidation, auto-stale, auto-prune.
    """

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
        quick: bool = False,             # v0.7.0: skip expensive dedup
        auto_stale: bool = False,        # v0.9.0: bump stale memories to priority 5
        auto_prune: bool = False,        # v0.9.0: delete priority-5 stale memories
    ) -> dict:
        """Generate a consolidation report. Read-only by default.

        v0.7.0: quick=True skips O(n²) duplicate detection.
        v0.9.0: auto_stale/auto_prune enable lifecycle management.

        Returns:
            report_id: str
            generated_at: str (ISO timestamp)
            scope: str
            points_scanned: int
            proposals: list[dict] — each a suggested action
            total_proposals: int
            lifecycle: dict — auto-stale/prune results (if enabled)
        """
        report_id = _generate_report_id()
        now = datetime.now(timezone.utc).isoformat()

        proposals = []
        total_scanned = 0

        if scope in ("memory", "both"):
            mem_proposals, mem_scanned = self._scan_collection(
                self._collection,
                max_points=max_points,
                max_groups=max_groups,
                include_examples=include_examples,
                quick=quick,
                auto_stale=auto_stale,
                auto_prune=auto_prune,
            )
            proposals.extend(mem_proposals)
            total_scanned += mem_scanned

        if scope in ("learning", "both") and self._learning:
            learn_proposals, learn_scanned = self._scan_collection(
                self._learning.collection_name,
                max_points=max_points,
                max_groups=max_groups,
                include_examples=include_examples,
                quick=quick,
            )
            proposals.extend(learn_proposals)
            total_scanned += learn_scanned

        # Save consolidation timestamp for incremental mode
        self._save_consolidation_time(now)

        result = {
            "report_id": report_id,
            "generated_at": now,
            "scope": scope,
            "points_scanned": total_scanned or max_points,
            "proposals": proposals,
            "total_proposals": len(proposals),
        }

        # Lifecycle proposals are in the proposals list, not a separate key
        return result

    def _scan_collection(
        self,
        collection_name: str,
        max_points: int,
        max_groups: int,
        include_examples: bool = False,
        quick: bool = False,
        auto_stale: bool = False,
        auto_prune: bool = False,
    ) -> tuple[list[dict], int]:
        """Scan a collection for duplicates and stale memories.

        Returns (proposals, points_scanned).
        """
        proposals = []

        try:
            points = self._fetch_all_points(collection_name, max_points)
            if not points:
                return proposals, 0

            # 1. Duplicate detection (skipped in quick mode)
            if not quick:
                dup_groups = self._find_duplicates(points, max_groups, include_examples)
                if dup_groups:
                    proposals.append({
                        "type": "duplicate_clusters",
                        "collection": collection_name,
                        "groups": dup_groups,
                        "count": len(dup_groups),
                        "points_considered": len(points),
                    })

            # 2. Stale detection
            stale = self._find_stale(points, include_examples)
            if stale:
                proposals.append({
                    "type": "stale_low_value",
                    "collection": collection_name,
                    "candidates": stale,
                    "count": len(stale),
                    "points_considered": len(points),
                })

            # 3. Quality warnings
            quality = self._find_quality_issues(points, include_examples)
            if quality:
                proposals.append({
                    "type": "quality_warnings",
                    "collection": collection_name,
                    "warnings": quality,
                    "count": len(quality),
                    "points_considered": len(points),
                })

            # 4. Correction pattern detection
            correction_patterns = self._find_correction_patterns(points, include_examples)
            if correction_patterns:
                proposals.append({
                    "type": "correction_patterns",
                    "collection": collection_name,
                    "patterns": correction_patterns,
                    "count": len(correction_patterns),
                    "points_considered": len(points),
                })

            # 5. Evolution suggestions
            evolution = self._find_evolution_suggestions(points, correction_patterns)
            if evolution:
                proposals.append({
                    "type": "evolution_suggestions",
                    "collection": collection_name,
                    "suggestions": evolution,
                    "count": len(evolution),
                    "points_considered": len(points),
                })

            # 6. Topic clustering (v0.8.0)
            if not quick:
                try:
                    from .clustering import TopicClustering
                    tc = TopicClustering(similarity_threshold=0.75)
                    clusters = tc.find_clusters(points)
                    if clusters:
                        proposals.append({
                            "type": "topic_clusters",
                            "collection": collection_name,
                            "clusters": clusters[:20],
                            "count": len(clusters),
                            "points_considered": len(points),
                        })
                except Exception as e:
                    logger.debug("Topic clustering skipped: %s", e)

            # 7. Auto-stale / auto-prune (v0.9.0)
            if auto_stale or auto_prune:
                lifecycle = self._apply_lifecycle(
                    points, collection_name,
                    auto_stale=auto_stale,
                    auto_prune=auto_prune,
                )
                if lifecycle:
                    proposals.append(lifecycle)

            # 8. Stats summary (always included)
            stats = self._compute_stats(points)
            proposals.append({
                "type": "stats",
                "collection": collection_name,
                "points_considered": len(points),
                **stats,
            })

        except Exception as e:
            logger.warning("Consolidation scan failed for %s: %s", collection_name, e)

        return proposals, len(points) if points else 0

    # ── internals ─────────────────────────────────────────────────────────

    def _fetch_all_points(self, collection_name: str, limit: int) -> list[dict]:
        """Scroll all points from a collection."""
        try:
            all_results = []
            offset = None
            page_size = min(limit, 500) if limit > 0 else 500

            while True:
                results, next_offset = self._store.client.scroll(
                    collection_name=collection_name,
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                all_results.extend(results)

                if not next_offset or not results:
                    break
                if limit > 0 and len(all_results) >= limit:
                    all_results = all_results[:limit]
                    break
                offset = next_offset

            points = []
            for r in all_results:
                p = {
                    "id": str(r.id),
                    "content": r.payload.get("content", "") or "",
                    "category": r.payload.get("category", ""),
                    "tags": r.payload.get("tags", []),
                    "version": int(r.payload.get("version", 1)),
                    "created_at": r.payload.get("created_at", ""),
                    "updated_at": r.payload.get("updated_at", r.payload.get("created_at", "")),
                    "priority": int(r.payload.get("priority", 3)),
                    "importance": int(r.payload.get("importance", 5)),  # legacy fallback
                    "source_type": r.payload.get("source_type", ""),
                    "origin": r.payload.get("origin", ""),
                    "evolved_from": r.payload.get("evolved_from", ""),
                    "vector": r.vector,
                }
                points.append(p)
            return points
        except Exception:
            return []

    def _find_duplicates(
        self, points: list[dict], max_groups: int, include_examples: bool
    ) -> list[dict]:
        """Find groups of semantically similar points.

        v0.9.0: Uses numpy batch cosine for O(n²) performance.
        """
        if len(points) < 2:
            return []

        # Extract vectors for batch computation
        vectors = []
        valid_indices = []
        for i, p in enumerate(points):
            v = p.get("vector")
            if v:
                vectors.append(v)
                valid_indices.append(i)

        if len(vectors) < 2:
            return []

        # Batch compute all pairwise similarities
        sim_matrix = _batch_cosine_similarity(vectors)

        # Find groups via connected components
        groups = []
        seen = set()

        for idx_i, i in enumerate(valid_indices):
            if i in seen:
                continue
            group = [points[i]]

            for idx_j, j in enumerate(valid_indices):
                if j <= i or j in seen:
                    continue
                if idx_j < len(sim_matrix[idx_i]) and sim_matrix[idx_i][idx_j] >= DUPLICATE_THRESHOLD:
                    group.append(points[j])
                    seen.add(j)

            if len(group) >= 2:
                seen.add(i)
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
            # Use updated_at (or fallback to created_at) for staleness check
            created = p.get("updated_at") or p.get("created_at", "")
            try:
                created_dt = datetime.fromisoformat(created)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            category = p.get("category", "")

            # Skip conversation category — always stale but handled separately
            if category in STALE_CATEGORIES_TO_SKIP:
                continue

            # Use priority (1=highest, 5=lowest). Fall back to importance for legacy data.
            priority = int(p.get("priority", 0))
            if priority == 0:
                # Legacy point: derive from importance (higher importance = higher priority)
                importance = int(p.get("importance", 5))
                priority = max(1, 6 - importance)  # importance 5→priority 1, importance 1→priority 5

            # Low priority (4-5) + stale = candidate
            if created_dt < cutoff and priority >= MIN_PRIORITY_FOR_STALE:
                stale.append({
                    "id": p["id"],
                    "content_preview": p["content"][:80],
                    "category": category,
                    "tags": p.get("tags", []),
                    "version": int(p.get("version", 1)),
                    "priority": priority,
                    "importance": int(p.get("importance", 5)),
                    "created_at": p["created_at"],
                    "updated_at": p.get("updated_at", ""),
                    "age_days": (datetime.now(timezone.utc) - created_dt).days,
                })

        return sorted(stale, key=lambda x: x["age_days"], reverse=True)[:50]

    def _find_quality_issues(
        self, points: list[dict], include_examples: bool
    ) -> list[dict]:
        """Find potential quality issues (very short content, possible secrets)."""
        warnings = []

        # More specific patterns to reduce false positives
        SECRET_PATTERNS = [
            r'\bsk-[a-zA-Z]{2,20}-[a-zA-Z0-9]{10,}\b',  # OpenAI-style keys
            r'\bapi[_-]?key\s*[=:]\s*["\']?[a-zA-Z0-9]{16,}',  # api_key=xxx (with value)
            r'\btoken\s*[=:]\s*["\']?[a-zA-Z0-9]{20,}',  # token=xxx (with value)
            r'\bpassword\s*[=:]\s*["\']?[^\s"\']{8,}',  # password=xxx (with value)
            r'-----BEGIN [A-Z ]+ PRIVATE KEY-----',  # PEM private keys
            r'\bAKIA[A-Z0-9]{16}\b',  # AWS access key IDs
            r'\bghp_[a-zA-Z0-9]{36}\b',  # GitHub personal access tokens
        ]
        for p in points:
            content = p.get("content", "")
            issues = []

            # Very short / low-quality content
            if len(content) < 20 and p.get("source_type") != "file":
                issues.append("very_short_content")

            # Possible secrets — use regex matching
            for pattern in SECRET_PATTERNS:
                if re.search(pattern, content):
                    issues.append(f"possible_secret:{pattern[:30]}")
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

    def _find_correction_patterns(
        self, points: list[dict], include_examples: bool
    ) -> list[dict]:
        """Find clusters of corrections by topic — suggests skill extraction.

        When N+ corrections share similar tags or content themes, they indicate
        a recurring pain point that should be extracted into a skill.
        """
        corrections = [
            p for p in points
            if p.get("category") == "correction" or p.get("origin") == "user_correction"
        ]

        if len(corrections) < 2:
            return []

        # Group by tags
        tag_groups: dict[str, list[dict]] = {}
        ungrouped: list[dict] = []

        for c in corrections:
            tags = c.get("tags", [])
            if tags:
                # Use first non-empty tag as group key
                key = tags[0] if tags else "ungrouped"
                tag_groups.setdefault(key, []).append(c)
            else:
                ungrouped.append(c)

        patterns = []
        # Tag-based groups with enough corrections
        for tag, group in sorted(tag_groups.items(), key=lambda x: -len(x[1])):
            if len(group) >= 2:
                pattern = {
                    "topic": tag,
                    "correction_count": len(group),
                    "content_previews": [c["content"][:80] for c in group[:5]],
                    "suggests_skill": len(group) >= 3,
                }
                if include_examples:
                    pattern["examples"] = [c["content"][:200] for c in group[:3]]
                patterns.append(pattern)

        return patterns[:10]

    def _find_evolution_suggestions(
        self, points: list[dict], correction_patterns: list[dict]
    ) -> list[dict]:
        """Generate evolution suggestions based on memory analysis.

        Suggests skills, cleanup actions, and improvements based on patterns.
        """
        suggestions = []

        # 1. Skill extraction from correction clusters
        for pattern in correction_patterns:
            if pattern.get("suggests_skill"):
                suggestions.append({
                    "type": "skill_extraction",
                    "topic": pattern["topic"],
                    "reason": f"{pattern['correction_count']} corrections on same topic — "
                              "extract into a skill to prevent recurrence.",
                    "priority": "high",
                })

        # 2. Category imbalance
        cat_counts: dict[str, int] = {}
        for p in points:
            cat = p.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        total = len(points)
        conversation_pct = cat_counts.get("conversation", 0) / max(total, 1)
        if conversation_pct > 0.5:
            suggestions.append({
                "type": "cleanup",
                "topic": "conversation_overflow",
                "reason": f"Conversation memories are {conversation_pct:.0%} of total ({cat_counts.get('conversation', 0)}/{total}). "
                          "Consider running consolidation cleanup.",
                "priority": "medium",
            })

        # 3. Priority distribution
        high_priority = sum(1 for p in points if int(p.get("priority", 3)) <= 2)
        if high_priority > 50:
            suggestions.append({
                "type": "review",
                "topic": "priority_inflation",
                "reason": f"{high_priority} memories have priority 1-2. "
                          "Review if all truly need high priority — too many high-priority items "
                          "reduces the effectiveness of priority boosting.",
                "priority": "low",
            })

        # 4. Low-correction health check
        correction_count = cat_counts.get("correction", 0)
        if correction_count == 0 and total > 50:
            suggestions.append({
                "type": "health",
                "topic": "no_corrections",
                "reason": "No correction memories found with >50 total memories. "
                          "Either the agent is perfect (unlikely) or corrections aren't being tagged. "
                          "Encourage tagging user corrections as category='correction'.",
                "priority": "low",
            })

        return suggestions

    def _compute_stats(self, points: list[dict]) -> dict:
        """Compute summary statistics for scanned points."""
        if not points:
            return {"total": 0}

        cat_counts: dict[str, int] = {}
        priority_counts: dict[int, int] = {}
        origin_counts: dict[str, int] = {}
        evolved_count = 0

        for p in points:
            cat = p.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            pri = int(p.get("priority", 3))
            priority_counts[pri] = priority_counts.get(pri, 0) + 1
            orig = p.get("origin", "unknown")
            origin_counts[orig] = origin_counts.get(orig, 0) + 1
            if p.get("evolved_from"):
                evolved_count += 1

        return {
            "total": len(points),
            "category_distribution": cat_counts,
            "priority_distribution": priority_counts,
            "origin_distribution": origin_counts,
            "evolved_memories": evolved_count,
        }

    # ── Incremental consolidation (v0.9.0) ───────────────────────────────

    def _get_last_consolidation_time(self) -> str:
        """Fetch last consolidation timestamp from metadata point."""
        try:
            meta = self._store.get_metadata_point(CONSOLIDATION_METADATA_ID)
            if meta and meta.get("last_consolidation"):
                return meta["last_consolidation"]
        except Exception:
            pass
        # Return epoch if no metadata found
        return "1970-01-01T00:00:00+00:00"

    def _save_consolidation_time(self, timestamp: str) -> None:
        """Store consolidation timestamp in metadata point."""
        try:
            self._store.upsert_metadata_point(
                CONSOLIDATION_METADATA_ID,
                payload={
                    "type": "consolidation_metadata",
                    "last_consolidation": timestamp,
                    "updated_at": timestamp,
                },
            )
        except Exception as e:
            logger.debug("Failed to save consolidation time: %s", e)

    # ── Memory lifecycle (v0.9.0) ────────────────────────────────────────

    def _apply_lifecycle(
        self,
        points: list[dict],
        collection_name: str,
        auto_stale: bool = False,
        auto_prune: bool = False,
    ) -> dict | None:
        """Apply auto-stale and auto-prune lifecycle management.

        auto_stale: bump stale low-priority memories to priority 5
        auto_prune: DELETE priority-5 memories older than prune_days

        Returns lifecycle proposal dict or None if nothing to do.
        """
        result = {
            "type": "lifecycle_management",
            "collection": collection_name,
            "auto_stale": {"enabled": auto_stale, "affected": 0, "ids": []},
            "auto_prune": {"enabled": auto_prune, "affected": 0, "ids": []},
        }

        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(days=AUTO_STALE_DAYS)
        prune_cutoff = now - timedelta(days=AUTO_PRUNE_DAYS)

        for p in points:
            updated = p.get("updated_at") or p.get("created_at", "")
            try:
                updated_dt = datetime.fromisoformat(updated)
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            priority = int(p.get("priority", 3))

            # Auto-stale: priority >= 4, older than stale_days → bump to 5
            if auto_stale and priority >= 4 and updated_dt < stale_cutoff:
                try:
                    ok = self._store.update_payload(p["id"], {
                        "priority": 5,
                        "auto_stale_at": now.isoformat(),
                    })
                    if ok:
                        result["auto_stale"]["affected"] += 1
                        result["auto_stale"]["ids"].append(p["id"])
                        logger.info(
                            "Auto-stale: bumped memory %s to priority 5 (was %d, age=%d days)",
                            p["id"], priority, (now - updated_dt).days,
                        )
                except Exception as e:
                    logger.warning("Auto-stale failed for %s: %s", p["id"], e)

            # Auto-prune: priority == 5, older than prune_days → DELETE
            if auto_prune and priority == 5 and updated_dt < prune_cutoff:
                try:
                    ok = self._store.delete(p["id"])
                    if ok:
                        result["auto_prune"]["affected"] += 1
                        result["auto_prune"]["ids"].append(p["id"])
                        logger.warning(
                            "Auto-prune: DELETED memory %s (priority=%d, age=%d days): %s",
                            p["id"], priority, (now - updated_dt).days,
                            p.get("content", "")[:80],
                        )
                except Exception as e:
                    logger.warning("Auto-prune failed for %s: %s", p["id"], e)

        # Truncate id lists for report readability
        result["auto_stale"]["ids"] = result["auto_stale"]["ids"][:20]
        result["auto_prune"]["ids"] = result["auto_prune"]["ids"][:20]

        if result["auto_stale"]["affected"] == 0 and result["auto_prune"]["affected"] == 0:
            return None

        return result


# ── math helpers ─────────────────────────────────────────────────────────────


def _generate_report_id() -> str:
    """Generate a unique report ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(ts.encode()).hexdigest()[:6]
    return f"report-{ts}-{suffix}"


def _redact_secrets(text: str) -> str:
    """Redact common secret patterns from text for safe preview."""
    text = re.sub(r'(sk-[a-zA-Z0-9]{10,})', '[REDACTED_KEY]', text)
    text = re.sub(r'(api_key[=:]\s*["\']?)([^"\'\s]+)', r'\1[REDACTED]', text)
    text = re.sub(r'-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----',
                   '[REDACTED_PRIVATE_KEY]', text, flags=re.DOTALL)
    return text

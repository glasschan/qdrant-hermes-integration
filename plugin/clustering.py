"""Topic clustering for Qdrant memory plugin (v0.8.0).

Uses cosine similarity to find topic clusters via connected components.
Each cluster: member memories, average similarity, auto-generated topic label.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from .consolidation import _cosine_similarity, _batch_cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_MIN_CLUSTER_SIZE = 2


class TopicClustering:
    """Find topic clusters among memories using connected components."""

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def find_clusters(self, points: list[dict]) -> list[dict]:
        """Find topic clusters using connected components of similar memories.

        Args:
            points: list of point dicts, each with 'id', 'content', 'tags', 'vector'.

        Returns:
            list of cluster dicts, sorted by size (largest first):
                - size: number of members
                - avg_similarity: average pairwise cosine similarity
                - topic_label: auto-generated label
                - member_ids: list of point IDs
                - content_previews: first 80 chars of each member's content
        """
        if len(points) < self.min_cluster_size:
            return []

        # Extract valid vectors
        vectors = []
        valid_points = []
        for p in points:
            v = p.get("vector")
            if v and p.get("content"):
                vectors.append(v)
                valid_points.append(p)

        if len(vectors) < self.min_cluster_size:
            return []

        # Batch compute pairwise similarities
        sim_matrix = _batch_cosine_similarity(vectors)

        # Find connected components using Union-Find
        n = len(valid_points)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if i < len(sim_matrix) and j < len(sim_matrix[i]):
                    if sim_matrix[i][j] >= self.similarity_threshold:
                        union(i, j)

        # Group by component
        components: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            components.setdefault(root, []).append(i)

        # Build cluster results
        clusters = []
        for indices in components.values():
            if len(indices) < self.min_cluster_size:
                continue

            members = [valid_points[i] for i in indices]

            # Compute average similarity within cluster
            member_vecs = [m.get("vector") for m in members if m.get("vector")]
            if len(member_vecs) >= 2:
                from .consolidation import _avg_group_similarity
                avg_sim = _avg_group_similarity(members)
            else:
                avg_sim = 0.0

            # Auto-generate topic label
            topic_label = self._generate_label(members)

            cluster = {
                "size": len(members),
                "avg_similarity": round(avg_sim, 4),
                "topic_label": topic_label,
                "member_ids": [m["id"] for m in members],
                "content_previews": [
                    m.get("content", "")[:80] for m in members[:5]
                ],
                "categories": list(set(
                    m.get("category", "") for m in members if m.get("category")
                )),
            }
            clusters.append(cluster)

        # Sort by size (largest first)
        clusters.sort(key=lambda c: c["size"], reverse=True)
        return clusters

    def _generate_label(self, members: list[dict]) -> str:
        """Auto-generate a topic label from cluster members.

        Strategy:
        1. Use most common tag if tags exist
        2. Fall back to most common category
        3. Fall back to first few words of first member
        """
        # Try tags first
        all_tags = []
        for m in members:
            tags = m.get("tags", [])
            if tags:
                all_tags.extend(tags)

        if all_tags:
            counter = Counter(all_tags)
            most_common = counter.most_common(3)
            label_parts = [tag for tag, _ in most_common]
            return " / ".join(label_parts)

        # Try categories
        categories = [
            m.get("category", "") for m in members if m.get("category")
        ]
        if categories:
            counter = Counter(categories)
            return counter.most_common(1)[0][0]

        # Fall back to first few words
        first_content = members[0].get("content", "")
        if first_content:
            words = first_content.split()[:5]
            return " ".join(words) + "..."

        return "unnamed_topic"

    def store_cluster_metadata(
        self, store, clusters: list[dict], user_id: str, agent_id: str
    ) -> int:
        """Store cluster metadata as special points in Qdrant.

        Each cluster gets a point with category='topic_summary'.
        Returns count of clusters stored.
        """
        from datetime import datetime, timezone
        import uuid

        stored = 0
        now = datetime.now(timezone.utc).isoformat()

        for cluster in clusters:
            try:
                point_id = str(uuid.uuid4())
                payload = {
                    "content": f"Topic cluster: {cluster['topic_label']} "
                               f"({cluster['size']} members, avg_sim={cluster['avg_similarity']})",
                    "category": "topic_summary",
                    "source_type": "cluster_metadata",
                    "member_ids": cluster["member_ids"][:50],
                    "topic_label": cluster["topic_label"],
                    "avg_similarity": cluster["avg_similarity"],
                    "cluster_size": cluster["size"],
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "created_at": now,
                    "updated_at": now,
                    "priority": 5,  # Low priority — metadata, not real memory
                }
                # Use the embed function from store's config
                from .embeddings import embed
                vector = embed(
                    [payload["content"]], store._config
                )[0]

                store.client.upsert(
                    collection_name=store.collection,
                    points=[
                        store._models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    ],
                )
                stored += 1
            except Exception as e:
                logger.debug("Failed to store cluster metadata: %s", e)

        logger.info("Stored %d cluster metadata points", stored)
        return stored

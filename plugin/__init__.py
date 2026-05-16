"""Qdrant vector memory plugin — plug-and-play Hermes memory provider.

Modular architecture (Lego-style):
  config.py      — constants + env-var loading
  embeddings.py  — OpenAI-compatible embedding client
  store.py       — QdrantStore (single-collection CRUD wrapper)
  schemas.py     — all 10 tool schemas
  indexer.py     — FileIndexer (directory indexing + manifest sync)
  learning.py    — LearningStore (procedural lessons, separate collection)
  consolidation.py — ConsolidationEngine (report-only dedup/stale/quality)
  provider.py    — QdrantMemoryProvider (wires everything together)

One-line install:
  cp -r plugin/ ~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant/
  hermes config set memory.provider hermes-memory-qdrant
"""

from __future__ import annotations

from .provider import QdrantMemoryProvider


def register(ctx) -> None:
    """Register the Qdrant memory provider plugin."""
    ctx.register_memory_provider(QdrantMemoryProvider())

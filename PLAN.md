# Qdrant Plugin v0.5.0 — Bug Fixes & Improvements Plan

> **For Hermes:** Execute this plan task-by-task. Each task is self-contained.

**Goal:** Fix all critical and high-priority bugs found in code review, plus medium-priority improvements.

**Architecture:** Minimal changes — fix bugs in-place, no architectural changes.

**Version:** v0.4.1 → v0.5.0

---

## Critical Bugs

### Task 1: Fix dedup vector update (store.py:231-258)
**Bug:** `set_payload()` updates payload but NOT the vector. After dedup, vector is stale.
**Fix:** Replace `set_payload()` with `upsert()` using the new vector.

### Task 2: Add pagination to `get_all()` (store.py:180-208)
**Bug:** Only fetches first 100 points. Silently drops data beyond 100.
**Fix:** Use scroll-based pagination to fetch all points.

### Task 3: Delete dead `learning.py`
**Bug:** 258 lines of dead code. Provider passes `learning_store=None`.
**Fix:** Delete the file.

## High Priority

### Task 4: Add retry logic to `embed()` (embeddings.py)
**Bug:** Single embedding failure crashes the operation. No retry, no fallback.
**Fix:** Add exponential backoff retry (3 attempts).

### Task 5: Add embedding dimension validation (embeddings.py)
**Bug:** No check that returned vectors match VECTOR_DIM. Dimension mismatch causes silent errors.
**Fix:** Validate dimensions after each embedding call.

### Task 6: Fix CLI status tool count (cli.py:80)
**Bug:** Says "10 tools" but plugin only has 6.
**Fix:** Correct the tool count.

## Medium Priority

### Task 7: Add `_client` accessor to QdrantStore (store.py)
**Bug:** Indexer, consolidation access `self._store._client` directly (breaks encapsulation).
**Fix:** Add `client` property to QdrantStore.

### Task 8: Log warnings instead of silently swallowing exceptions (config.py, store.py, indexer.py)
**Bug:** `except Exception: pass` hides real errors.
**Fix:** Add `logger.warning()` in caught exceptions.

### Task 9: Fix `_load_manifest` to skip vectors (indexer.py:256-268)
**Bug:** Loads vectors (2048 floats each) for manifest but never uses them.
**Fix:** Add `with_vectors=False`.

### Task 10: Fix `_find_stale_ids` to batch queries (indexer.py:279-308)
**Bug:** N sequential scroll queries for N files. O(N) round trips.
**Fix:** Batch into single query with `should` filter conditions.

### Task 11: Improve secret detection patterns (consolidation.py:279)
**Bug:** Simple substring matching causes false positives ("sk-learn", "password policy").
**Fix:** Improve patterns to be more specific.

### Task 12: Remove unused `_max_chunk_tokens` config (indexer.py:113)
**Bug:** Configured but never used. Chunking is character-based, not token-based.
**Fix:** Remove the unused config variable.

### Task 13: Add tag filtering to `qdrant_search` (store.py, schemas.py)
**Missing:** Tags are stored but search can't filter by them.
**Fix:** Add optional `tags` filter to search method and schema.

### Task 14: Bump version to v0.5.0
**Fix:** Update VERSION, plugin.yaml, and cli.py tool count.

---

## Execution Order
Tasks 1-14 in sequence. Each task is a single commit.

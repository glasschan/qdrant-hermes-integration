# Qdrant Integration Improvement Plan

Branch: `feature/dry-run-indexing-learning-consolidation`

## Phase 3: Dry-Run Safety on qdrant_forget
- Add `dry_run` parameter to FORGET_SCHEMA (default: true)
- In handle_tool_call, when dry_run=true: return what WOULD be deleted without deleting
- When dry_run=false: actually delete (user must explicitly opt-in)

## Phase 4: File Indexing
- New tool: `qdrant_index` — index .md/.txt files into Qdrant
- New file: `plugin/indexer.py` — FileIndexer class
- Features:
  - Index single files or directories
  - Manifest sync (detect changed/deleted files via hash)
  - Dry-run first (preview chunks before upserting)
  - Configurable extensions, exclude dirs, max files
  - Payload: source_type="file", file_path, heading, chunk_index

## Phase 5: Learning Collection
- New tools: `qdrant_learning_store`, `qdrant_learning_search`, `qdrant_learning_preview`
- New file: `plugin/learning.py` — LearningStore class
- Separate Qdrant collection: `<collection_name>_learnings`
- Structured fields: lesson, learning_type, trigger, mistake, correction, evidence
- Auto-extract disabled by default (manual/gated only)

## Phase 6: Basic Consolidation
- New tool: `qdrant_consolidate` — report-only, never mutates
- New file: `plugin/consolidation.py`
- Features:
  - Duplicate detection (cosine similarity threshold)
  - Stale memory detection (age-based)
  - Report output only — no automatic deletion/merge

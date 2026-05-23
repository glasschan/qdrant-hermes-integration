---
name: hermes-memory-qdrant
description: "Complete guide: add Qdrant vector memory to any Hermes Agent instance — plugin setup, config, migration, and troubleshooting."
version: 0.2.0
metadata:
  hermes:
    tags: [memory, qdrant, vector, plugin]
---

# Hermes Qdrant Memory Plugin — Setup Guide

Add semantic long-term memory to any Hermes instance using Qdrant + OpenAI-compatible embeddings.

## ⚠️ CRITICAL: IMMUTABLE SAFETY RULES ⚠️

### 🚫 Rule 1 — NEVER touch another agent's collection
**Each agent only operates on its OWN collection.** The collection name is hard-scoped to `self._collection`. The plugin physically cannot reference any other collection name — every read/write/search/delete goes through `self._collection`.

### 🚫 Rule 2 — NEVER delete any collection
**The plugin contains zero `delete_collection()` calls.** The only delete operation is `qdrant_forget`, which deletes individual memory points (by point ID) within the agent's own collection. Entire collections can never be dropped.

### ✅ Rule 3 — Auto-scoped collection names
If `QDRANT_COLLECTION` is not set, the plugin auto-generates:
```
hermes_memories_<hostname>_<profile>
```
Each Hermes instance (hostname + profile combo) automatically gets its own namespace. No config needed.

If you set `QDRANT_COLLECTION` manually, make it unique:
```ini
QDRANT_COLLECTION=hermes_memories_prod_server_a
QDRANT_COLLECTION=hermes_memories_dev_laptop
```

## Prerequisites

- Qdrant server running (local Docker, self-hosted, or Qdrant Cloud)
- An OpenAI-compatible embedding API endpoint + API key
- Hermes Agent installed (`hermes` CLI available)

## Installation

### One-Liner
```bash
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash
```

### Manual
```bash
# 1. Copy plugin to user-installed plugins dir
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/

# 2. Install dep
pip install qdrant-client

# 3. Configure
hermes config set memory.provider hermes-memory-qdrant

# 4. Add env vars to ~/.hermes/.env
cat >> ~/.hermes/.env << 'EOF'
QDRANT_URL=http://localhost:6333
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=your-embedding-model
EOF

# 5. Restart
hermes gateway restart
```

### Updating
```bash
# Check for update and upgrade
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --update

# Force reinstall
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --force
```

## Env Vars Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | ✓* | http://localhost:6333 | Qdrant server URL |
| `QDRANT_API_KEY` | — | — | Qdrant API key |
| `QDRANT_COLLECTION` | — | auto-generated† | Per-agent namespace. Plugin never touches other collections. |
| `EMBEDDING_BASE_URL` | ✓ | — | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_API_KEY` | ✓ | — | Embedding API key |
| `EMBEDDING_MODEL` | — | doubao-embedding-vision | Embedding model name |

*> ✓ means recommended, URL defaults to localhost for local dev*
*† Auto-generated as `hermes_memories_<hostname>_<profile>` — unique per machine + profile*

## Tools (10)

| # | Tool | What it does |
|---|------|-------------|
| 1 | `qdrant_profile` | Get all stored memories |
| 2 | `qdrant_search` | Semantic search by meaning |
| 3 | `qdrant_remember` | Store a fact (preference/fact/decision/goal/instruction) |
| 4 | `qdrant_forget` | Delete by point ID — dry-run first (safe default) |
| 5 | `qdrant_index` | Index .md/.txt files with manifest sync |
| 6 | `qdrant_consolidate` | Report-only duplicate/stale/quality detection |
| 7 | `qdrant_learning_store` | Store procedural lessons (gated/manual) |
| 8 | `qdrant_learning_search` | Search procedural learnings |
| 9 | `qdrant_learning_preview` | Preview pending learning candidates |
| 10 | `qdrant_learning_approve` | Approve and store a candidate |

## CLI Commands

If the plugin has `plugin/cli.py`, these are available:
```bash
hermes memory-qdrant status    # Show active config
hermes memory-qdrant stats     # Memory count, learning count
hermes memory-qdrant version   # Current + latest available
hermes memory-qdrant update    # Check for update and upgrade
hermes memory-qdrant flush     # Clear conversation memories (keep facts)
```

## Plugin Architecture

```
~/.hermes/plugins/hermes-memory-qdrant/   ← User-installed path
├── plugin.yaml      # Plugin metadata + pip deps
├── VERSION          # Version file (plaintext)
├── __init__.py      # Entry — import + register()
├── config.py        # Env var loading + constants
├── embeddings.py    # OpenAI-compatible embedding client
├── store.py         # QdrantStore — single-collection CRUD
├── schemas.py       # All 10 tool JSON schemas
├── provider.py      # QdrantMemoryProvider — wires everything
├── indexer.py       # FileIndexer — .md/.txt + manifest sync
├── learning.py      # LearningStore — procedural lessons
├── consolidation.py # ConsolidationEngine — report-only
├── cli.py           # CLI subcommands (optional)
```

**Total: ~1,900 lines across 9+1 modules.**

## Troubleshooting

| Problem | Likely Fix |
|---------|------------|
| `UserWarning: Api key with insecure connection` | Qdrant runs on HTTP — safe for local dev, add `export PYTHONWARNINGS="ignore::UserWarning"` to `.env` |
| `ModuleNotFoundError: No module named 'qdrant_client'` | Run `pip install qdrant-client` |
| Memory not prefetching | Run `/reset` or start a new `hermes` session |
| Embedding API returns 401 | Check `EMBEDDING_API_KEY` in `.env` |
| Plugin not discovered | Check it's at `~/.hermes/plugins/hermes-memory-qdrant/__init__.py`
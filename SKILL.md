---
name: hermes-memory-qdrant
description: "Complete guide: add Qdrant vector memory to any Hermes Agent instance — plugin setup, config, migration, and troubleshooting."
version: 0.1.0
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

### Enforcement in Code
```python
# Every operation uses self._collection — the instance variable set once on init
def search(self, ...):    client.query_points(collection_name=self._collection, ...)
def get_all(self, ...):   client.scroll(collection_name=self._collection, ...)
def add(self, ...):       client.upsert(collection_name=self._collection, ...)
def delete(self, ...):    client.delete(collection_name=self._collection, ...)
#                     ↑↑↑ NEVER a variable, NEVER a parameter — always self._collection
```

## Overview

```
User message → Embedding API → Qdrant search → Relevant memories injected
                      ↕
         Qdrant collection (scoped per agent — never shared!)
```

## Prerequisites

- Qdrant server running (local Docker, self-hosted, or Qdrant Cloud)
- An OpenAI-compatible embedding API endpoint + API key
- Hermes Agent installed (`hermes` CLI available)

## Step 1: Create the Plugin

Copy `__init__.py` and `plugin.yaml` from this pack to:
```
~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant/
```

## Step 2: Install Dependencies

```bash
cd ~/.hermes/hermes-agent
uv pip install qdrant-client   # or: pip install qdrant-client
```

Or if using the built-in venv:
```bash
VIRTUAL_ENV=~/.hermes/hermes-agent/venv uv pip install qdrant-client
```

## Step 3: Configure Hermes

```bash
hermes config set memory.provider hermes-memory-qdrant
```

Add to `~/.hermes/.env`:
```ini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-key
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=your-embedding-model

# Optional: set a unique name per deployment (auto-generated if empty)
QDRANT_COLLECTION=hermes_memories_my_project
```

## Step 4: Verify

```bash
hermes doctor --fix
hermes chat -q "使用 qdrant_remember 記低：This is a test memory"
hermes chat -q "使用 qdrant_search 搜尋 'test memory'"
```

If the second session returns the stored memory, integration is complete.

## Env Vars Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | ✓* | http://localhost:6333 | Qdrant server URL |
| `QDRANT_API_KEY` | — | — | Qdrant API key |
| `QDRANT_COLLECTION` | — | auto-generated† | **Per-agent namespace. Plugin never touches other collections.** |
| `EMBEDDING_BASE_URL` | ✓ | — | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_API_KEY` | ✓ | — | Embedding API key |
| `EMBEDDING_MODEL` | — | doubao-embedding-vision | Embedding model name |

*> ✓ means recommended, URL defaults to localhost for local dev*
*† Auto-generated as `hermes_memories_<hostname>_<profile>` — unique per machine + profile*

## Tools

| Tool | Description |
|------|-------------|
| `qdrant_search` | Semantic search within own collection |
| `qdrant_remember` | Store a fact (preference/fact/decision/goal/instruction) within own collection |
| `qdrant_profile` | Get all memories within own collection |
| `qdrant_forget` | Delete a single point by ID within own collection — **never drops collections** |

## Architecture Note

qdrant-client v1.18+ uses `query_points()` not `search()` — the plugin uses the newer API.

## Troubleshooting

| Problem | Likely Fix |
|---------|------------|
| `UserWarning: Api key with insecure connection` | Qdrant runs on HTTP — safe for local dev, add `export PYTHONWARNINGS="ignore::UserWarning"` to `.env` |
| `ModuleNotFoundError: No module named 'qdrant_client'` | Run the pip install step |
| Memory not prefetching | Run `/reset` or start a new `hermes` session |
| Embedding API returns 401 | Check `EMBEDDING_API_KEY` in `.env` |

## Plugin Architecture

```
plugins/memory/hermes-memory-qdrant/
├── plugin.yaml      # Plugin metadata
├── __init__.py      # MemoryProvider implementation (~620 lines)
```

The `__init__.py` provides:
- `_QdrantStore` — thin wrapper around `qdrant-client` (self._collection, upsert, query_points, scroll, delete point only)
- `QdrantMemoryProvider` — implements Hermes's `MemoryProvider` ABC
- Tools: `qdrant_search`, `qdrant_remember`, `qdrant_profile`, `qdrant_forget`
- Auto-prefetch before each turn
- Auto-ingest after each turn (non-blocking thread)
- **Immutable safety**: collection hard-scoped to `self._collection`, zero delete_collection calls

---
name: hermes-qdrant-integration
description: "Plug-and-play Qdrant vector memory plugin for Hermes Agent — 6 tools, 10 modules, 58 tests, CLI commands. Install on any Hermes instance in 1 minute."
version: 3.0.0
author: Glass Chan + Paul
license: MIT
metadata:
  hermes:
    tags: [qdrant, memory, plugin, vector-db, hermes-agent, lego-architecture]
    related_skills: [hermes-memory-qdrant]
---

# Hermes Qdrant Integration Plugin

One-minute Qdrant vector memory for any Hermes Agent instance. Semantic long-term memory via local Qdrant + any OpenAI-compatible embedding API. **6 tools, 10 modules, 58 tests, 3 CLI commands.**

## Quick Install

```bash
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash
```

## Manual Install

```bash
# 1. Copy plugin
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/

# 2. Install dep
source ~/.hermes/hermes-agent/venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install qdrant-client

# 3. Configure
hermes config set memory.provider hermes-memory-qdrant

# 4. Add env vars to ~/.hermes/.env
cat >> ~/.hermes/.env << 'EOF'
QDRANT_URL=http://localhost:6333
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=your-embedding-model
EOF

# 5. Restart
hermes gateway restart
```

## Tools (6)

| Tool | Description |
|------|-------------|
| `qdrant_profile` | Get all stored memories |
| `qdrant_search` | Semantic search by meaning. Optional `recency_weight` (0.0-1.0) |
| `qdrant_remember` | Store a fact. Auto-dedup on same content. Optional `tags` array |
| `qdrant_forget` | Delete by point ID. Dry-run by default |
| `qdrant_index` | Index .md/.txt files/directories with manifest sync |
| `qdrant_consolidate` | Report-only duplicate/stale/quality detection |

## CLI Commands

```bash
hermes hermes-memory-qdrant status      # Show plugin status + env vars
hermes hermes-memory-qdrant version     # Current vs latest release
hermes hermes-memory-qdrant update      # Upgrade to latest from GitHub
```

## Architecture

```
plugin/
├── __init__.py         # Entry point + register() + self-healing
├── config.py           # Env var loading + constants
├── embeddings.py       # OpenAI-compatible embedding client
├── store.py            # QdrantStore — CRUD + dedup + recency
├── schemas.py          # All 6 tool JSON schemas
├── provider.py         # QdrantMemoryProvider — wires everything
├── indexer.py          # FileIndexer — directory indexing + manifest sync
├── consolidation.py    # ConsolidationEngine — read-only dedup/stale/quality
├── clustering.py       # Topic clustering — groups similar memories
├── cli.py              # CLI subcommands (status, version, update)
├── plugin.yaml         # Hermes plugin metadata
└── VERSION             # Plaintext version
```

~4,000 lines across 10 modules. 58 unit tests.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `hermes doctor` shows "plugin not found" | Run setup.sh or check plugin at `~/.hermes/plugins/hermes-memory-qdrant/` |
| `ModuleNotFoundError: No module named 'qdrant_client'` | `pip install qdrant-client` in Hermes venv |
| `No module named '_hermes_user_memory'` | Fixed — plugin self-heals at load time |
| Tools not showing up | Check `plugins.enabled` in config.yaml + `memory.provider` |
| Qdrant tools visible but not working | `hermes gateway restart` |
| Embedding API returns 401 | Check `EMBEDDING_API_KEY` in `~/.hermes/.env` |

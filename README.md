# Hermes Qdrant Memory Plugin

> Semantic long-term vector memory for Hermes Agent — local Qdrant + any OpenAI-compatible embedding API.
> One-minute install. Zero-config auto-scoping. Self-healing.

[![version](https://img.shields.io/badge/version-0.9.2-blue)](https://github.com/glasschan/qdrant-hermes-integration/releases)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why This Exists

Hermes has built-in memory (MEMORY.md / USER.md), but it's flat-file. No semantic search, no dedup, no multi-agent scoping. This plugin adds a proper vector memory backend — store facts, search by meaning, and never worry about duplicates.

## Features

- **Semantic search** — find memories by meaning, not keywords
- **Pre-save dedup** — same fact auto-updates instead of creating duplicates
- **Recency-weighted search** — blend freshness with semantic relevance (0.0–1.0)
- **Tags** — optional string arrays for filtered retrieval
- **Auto-scoped collections** — each Hermes instance + profile combo gets its own namespace
- **Read-only consolidation** — find duplicates, stale entries, quality issues without touching data
- **File indexing** — index .md/.txt files and directories with manifest sync
- **Self-healing** — automatically works around a known Hermes namespace bug, no patches needed
- **CLI tools** — `status`, `version`, `update` subcommands
- **Safe by default** — dry-run on destructive ops, circuit breaker on failures, zero collection-delete code

---

## Quick Install

```bash
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash
```

The setup script handles everything:
- Installs plugin files to `~/.hermes/plugins/hermes-memory-qdrant/`
- Installs `qdrant-client` in the Hermes venv
- Sets `memory.provider` in config.yaml
- Adds to `plugins.enabled`
- Prompts for env vars (URL, API keys, embedding model)

### Manual Install

```bash
# 1. Copy plugin
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/

# 2. Install dep
source ~/.hermes/hermes-agent/venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install qdrant-client

# 3. Configure
hermes config set memory.provider hermes-memory-qdrant

# 4. Enable plugin (add to plugins.enabled in config.yaml)
#     Use Python to avoid the JSON-string YAML bug in `hermes config set`
python3 -c "
import re
path = '$HOME/.hermes/config.yaml'
with open(path) as f: content = f.read()
match = re.search(r'enabled:\s*\n(\s+- .+\n?)*', content)
if match:
    block = match.group()
    if 'hermes-memory-qdrant' not in block:
        content = content.replace(block, block.rstrip() + '\n  - hermes-memory-qdrant\n')
        with open(path, 'w') as f: f.write(content)
else:
    content += '\nplugins:\n  enabled:\n  - hermes-memory-qdrant\n'
    with open(path, 'w') as f: f.write(content)
"

# 5. Add env vars to ~/.hermes/.env
cat >> ~/.hermes/.env << 'EOF'

# Qdrant Memory
QDRANT_URL=http://localhost:6333
EMBEDDING_BASE_URL=https://your-api.com/v1
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=doubao-embedding-vision
EOF

# 6. Restart
hermes gateway restart
```

### Prerequisites

- Hermes Agent installed
- Qdrant server running (`docker run -p 6333:6333 qdrant/qdrant`)
- OpenAI-compatible embedding API endpoint + key

---

## Updating

```bash
# Check for update and upgrade
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --update

# Force reinstall
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --force

# Via CLI
hermes hermes-memory-qdrant status
hermes hermes-memory-qdrant version
hermes hermes-memory-qdrant update
```

---

## Tools (6)

| Tool | Description |
|------|-------------|
| `qdrant_profile` | Retrieve all stored memories — preferences, facts, project context |
| `qdrant_search` | Semantic search by meaning. Optional `recency_weight` (0.0–1.0) to favor fresh results |
| `qdrant_remember` | Store a durable fact. Auto-dedup — same content updates existing entry. Optional `tags` array |
| `qdrant_forget` | Delete a memory by point ID. **Dry-run by default** — preview before deleting |
| `qdrant_index` | Index .md/.txt files or directories. **Dry-run by default** — supports manifest sync |
| `qdrant_consolidate` | Read-only report. Finds duplicates, stale memories, quality issues. **Never mutates data** |

---

## CLI Commands

```bash
hermes hermes-memory-qdrant status      # Show plugin status, version, env vars
hermes hermes-memory-qdrant version     # Current vs latest GitHub release
hermes hermes-memory-qdrant update      # Check for update and upgrade
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `QDRANT_URL` | — | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | — | — | Qdrant API key (for Qdrant Cloud) |
| `QDRANT_COLLECTION` | — | auto-generated | Per-agent collection name. Unique per machine + profile |
| `EMBEDDING_BASE_URL` | ✓ | — | OpenAI-compatible embedding endpoint |
| `EMBEDDING_API_KEY` | ✓ | — | API key for the embedding service |
| `EMBEDDING_MODEL` | — | `doubao-embedding-vision` | Embedding model name |
| `QDRANT_DEDUP_THRESHOLD` | — | `0.85` | Cosine similarity threshold for pre-save dedup |
| `QDRANT_DEDUP_ENABLED` | — | `true` | Enable/disable pre-save dedup |
| `QDRANT_AUTO_SYNC` | — | `false` | Auto-save user messages to memory |
| `QDRANT_RECENCY_WEIGHT` | — | `0.0` | Default recency weight for search (0.0–1.0) |

Auto-generated collection name format: `hermes_memories_<hostname>_<profile>`

---

## Self-Healing: Hermes Namespace Bug

When Hermes loads user-installed memory provider plugins, it uses the `_hermes_user_memory.<name>` namespace but **never registers `_hermes_user_memory` as a Python package** in `sys.modules`. This breaks relative imports (`from .schemas import ...` → `ModuleNotFoundError`).

**This plugin self-heals at load time:**
1. Detects the broken namespace from `__name__`
2. Registers the missing parent package in `sys.modules`
3. Strips half-loaded submodules and reloads via `importlib`

**No Hermes patches, no dual-path install, no config changes needed.** Single user-installed path works everywhere.

---

## Architecture

```
plugin/
├── __init__.py        # Entry point — self-healing + register()
├── config.py          # Env var loading + constants + memory hygiene
├── embeddings.py      # OpenAI-compatible embedding client
├── store.py           # QdrantStore — CRUD + pre-save dedup + recency search
├── schemas.py         # All 6 tool JSON schemas
├── provider.py        # QdrantMemoryProvider — wires everything together
├── indexer.py         # FileIndexer — directory indexing + manifest sync
├── consolidation.py   # ConsolidationEngine — read-only dedup/stale/quality
├── clustering.py      # Topic clustering — groups similar memories
├── cli.py             # CLI subcommands (status, version, update)
├── plugin.yaml        # Hermes plugin metadata + pip deps
└── VERSION            # Plaintext version
```

**~4,000 lines across 10 modules. 58 unit tests.** Each file self-contained, independently testable. Lego-style — swap any piece without touching the rest.

---

## Safety

| Rule | Enforcement |
|------|-------------|
| Never delete any Qdrant collection | Zero `delete_collection()` calls in codebase |
| Each agent = own collection | Hard-scoped to `self._collection` at init |
| Dry-run first | `qdrant_forget`, `qdrant_index` default `dry_run=true` |
| Consolidation is read-only | `qdrant_consolidate` finds issues, never mutates |
| Pre-save dedup | Auto-updates existing points instead of creating duplicates |
| Circuit breaker | Pauses on 5+ consecutive failures, auto-resumes after 120s |

---

## Memory Hygiene

Built-in accuracy-first design — every memory is stored because the agent will read it later:

- **Pre-save dedup** — same content auto-updates existing entry (no duplicates)
- **Auto-metadata** — every point gets `created_at`, `updated_at`, `version`
- **Category validation** — rejects unrecognized categories (allowed: `fact`, `decision`, `instruction`, `goal`, `preference`)
- **`sync_turn()` OFF by default** — no auto-saved conversation noise
- **Payload indexes** — on `category` + `updated_at` for fast filtered queries

---

## Verification

```bash
# Check provider status
hermes doctor --fix | grep "Memory Provider"
# Expected: ✓ hermes-memory-qdrant provider active

# Check tools
hermes chat -q "list all Qdrant tools you have access to"
# Expected: 6 tools (qdrant_profile, qdrant_search, qdrant_remember,
#            qdrant_forget, qdrant_index, qdrant_consolidate)

# Check version
hermes hermes-memory-qdrant status
```

---

## Repo Structure

```
hermes-qdrant-integration/
├── README.md           # ← this file
├── SKILL.md            # Full setup guide + troubleshooting
├── setup.sh            # One-liner installer (--update, --force flags)
├── pyproject.toml      # Test config (pytest)
├── plugin/
│   ├── __init__.py     # Self-healing entry point
│   ├── plugin.yaml     # Hermes plugin metadata
│   ├── VERSION         # Plaintext version
│   ├── config.py       # Config + constants
│   ├── embeddings.py   # Embedding client
│   ├── store.py        # Qdrant CRUD
│   ├── schemas.py      # Tool schemas
│   ├── provider.py     # MemoryProvider
│   ├── indexer.py      # File indexing
│   ├── consolidation.py # Memory consolidation
│   ├── clustering.py   # Topic clustering
│   └── cli.py          # CLI subcommands
├── tests/              # 58 unit tests
│   ├── conftest.py     # Hermes dependency mocks
│   ├── test_config.py
│   ├── test_consolidation.py
│   ├── test_indexer.py
│   └── test_store.py
└── references/         # Architecture docs, security checklist, deployment logs
```

---

## License

MIT — use it, modify it, ship it.

Built by [Glass Chan](https://github.com/glasschan) + Paul.

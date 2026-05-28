# Hermes Qdrant Memory — Lego Plugin v0.4.0

> **6 tools · 6 modules · 3 CLI commands**

Qdrant-backed persistent vector memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Semantic search over facts, file indexing, memory consolidation — all local-first.

**v0.3.0 highlights:**
- **Pre-save dedup** — same fact auto-updates instead of creating duplicates
- **Payload metadata** — every entry tracked with `version`, `created_at`, `updated_at`
- **Tags** — optional string arrays for filtered retrieval
- **Recency-weighted search** — blend freshness with semantic relevance
- **Auto-indexes** — payload indexes on `category` + `updated_at` for fast queries
- **Noise reduction** — `sync_turn()` OFF by default (no auto-saved garbage)

## ⚡ One-Liner Install

```bash
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash
```

Or manual (note: requires installation to **two** paths):

```bash
# 1. User path — for tool registration + CLI
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/

# 2. Bundled memory path — for memory provider discovery (plugins.memory namespace)
cp -r plugin/ ~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant/

# 3. Install dependency in Hermes venv
source ~/.hermes/hermes-agent/venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install qdrant-client

# 4. Configure
hermes config set memory.provider hermes-memory-qdrant

# 5. Enable plugin (for tool schemas)
#    Use Python to avoid the JSON-string YAML bug
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

# 6. Add env vars to ~/.hermes/.env
echo 'QDRANT_URL=http://localhost:6333' >> ~/.hermes/.env
echo 'EMBEDDING_BASE_URL=https://your-api.com/v1' >> ~/.hermes/.env
echo 'EMBEDDING_API_KEY=*** >> ~/.hermes/.env
echo 'EMBEDDING_MODEL=doubao-embedding-vision' >> ~/.hermes/.env

# 7. Restart
hermes gateway restart
```

## 🔄 Updating

```bash
# Check for update and upgrade
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --update

# Force reinstall (same version)
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --force

# Or via CLI (if cli.py is installed)
hermes hermes-memory-qdrant status
hermes hermes-memory-qdrant version
hermes hermes-memory-qdrant update
```

## 🧰 Tools (6)

| # | Tool | What it does |
|---|------|-------------|
| 1 | `qdrant_profile` | Get all stored memories with tags + version info |
| 2 | `qdrant_search` | Semantic search by meaning, optional `recency_weight` (0.0-1.0) |
| 3 | `qdrant_remember` | Store a fact — auto-dedup, optional `tags` array |
| 4 | `qdrant_forget` | Delete by point ID — **dry-run first** (safe default) |
| 5 | `qdrant_index` | Index .md/.txt files with manifest sync |
| 6 | `qdrant_consolidate` | Report-only duplicate/stale/quality detection |

## 🧱 Lego Architecture

```
plugin/
├── __init__.py       ( 25)  # entry — import + register()
├── cli.py            (222)  # CLI subcommands (status, version, update)
├── config.py         ( 56)  # env var loading + constants
├── embeddings.py     ( 30)  # OpenAI-compatible embedding client
├── store.py          (372)  # QdrantStore — CRUD + pre-save dedup + indexes
├── schemas.py        (147)  # all 6 tool JSON schemas
├── provider.py       (391)  # QdrantMemoryProvider — wires everything
├── indexer.py        (359)  # FileIndexer — .md/.txt + manifest sync
└── consolidation.py  (337)  # ConsolidationEngine — report-only
```

**Total: ~1,900 lines across 6 modules. Each file self-contained, independently testable. Swap any piece without touching the rest.**

## ⚠️ Safety Rules

| Rule | Enforcement |
|------|------------|
| Never delete any Qdrant collection | Zero `delete_collection()` calls in codebase |
| Each agent = own collection | Hard-scoped to `self._collection` at init |
| Dry-run first | `qdrant_forget`, `qdrant_index` default dry_run=true |
| Consolidation = read-only | `qdrant_consolidate` finds issues, NEVER mutates |
| Pre-save dedup | Auto-updates existing points instead of creating duplicates |

## 📦 Prerequisites

- Python 3.10+ with `qdrant-client`
- Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`)
- OpenAI-compatible embedding API endpoint + key

## 🚀 Deploy to Another Hermes Instance

From this repo:

```bash
# Clone the repo
git clone https://github.com/glasschan/qdrant-hermes-integration.git
cd hermes-qdrant-integration

# Run setup (interactive)
bash setup.sh

# Or manual deploy (dual path install required)
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/
cp -r plugin/ ~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant/
source ~/.hermes/hermes-agent/venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install qdrant-client
hermes config set memory.provider hermes-memory-qdrant
hermes gateway restart
```

From a remote Hermes (user downloads your repo):

```bash
# Their machine
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash
```

## 🧪 Verification

```bash
hermes doctor --fix
# Expected: ✓ hermes-memory-qdrant provider active

hermes chat -q "list all Qdrant tools you have access to"
# Expected: 6 tools listed
```

## 📄 Env Vars

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | No | — | Qdrant API key |
| `QDRANT_COLLECTION` | No | auto | Collection name (auto-generated if empty) |
| `EMBEDDING_BASE_URL` | **Yes** | — | OpenAI-compatible embedding endpoint |
| `EMBEDDING_API_KEY` | **Yes** | — | API key for embedding service |
| `EMBEDDING_MODEL` | No | `doubao-embedding-vision` | Embedding model name |
| `QDRANT_DEDUP_THRESHOLD` | No | `0.85` | Cosine similarity threshold for dedup |
| `QDRANT_DEDUP_ENABLED` | No | `true` | Enable/disable pre-save dedup |
| `QDRANT_AUTO_SYNC` | No | `false` | Auto-save user messages to memory |
| `QDRANT_RECENCY_WEIGHT` | No | `0.0` | Recency weight in search (0.0-1.0) |

## 📁 Full Repo

```
hermes-qdrant-integration/
├── README.md          # ← this file
├── SKILL.md           # Full setup guide + troubleshooting
├── PLAN.md            # Implementation plan
├── setup.sh           # Installer with --update/--force flags
└── plugin/
    ├── plugin.yaml    # Hermes plugin metadata (v0.4.0)
    ├── VERSION        # Plaintext version (v0.4.0)
    ├── cli.py         # CLI subcommands
    ├── __init__.py    # Entry point
    ├── config.py      # Config loading
    ├── embeddings.py  # Embedding client
    ├── store.py       # Qdrant CRUD + dedup
    ├── schemas.py     # Tool definitions
    ├── provider.py    # MemoryProvider impl
    ├── indexer.py     # File indexing
    └── consolidation.py  # Memory consolidation
```

## 📜 License

MIT — use it, modify it, ship it.
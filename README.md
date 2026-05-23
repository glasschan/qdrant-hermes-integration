# Hermes Qdrant Memory — Lego Plugin v0.2.0

> **10 tools · 10 modules · 3 CLI commands**

Qdrant-backed persistent vector memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Semantic search over facts, file indexing, procedural learning, and memory consolidation — all local-first.

## ⚡ One-Liner Install

```bash
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash
```

Or manual:

```bash
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/
hermes config set memory.provider hermes-memory-qdrant
hermes gateway restart
```

## 🔄 Updating

```bash
# Check for update and upgrade
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --update

# Force reinstall (same version)
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --force

# Or via CLI (if cli.py is installed)
hermes hermes-memory-qdrant --update
hermes hermes-memory-qdrant --check-version
hermes hermes-memory-qdrant --status
```

## 🧰 Tools (10)

| # | Tool | What it does |
|---|------|-------------|
| 1 | `qdrant_profile` | Get all stored memories |
| 2 | `qdrant_search` | Semantic search by meaning |
| 3 | `qdrant_remember` | Store a fact (preference/fact/decision/goal/instruction) |
| 4 | `qdrant_forget` | Delete by point ID — **dry-run first** (safe default) |
| 5 | `qdrant_index` | Index .md/.txt files with manifest sync |
| 6 | `qdrant_consolidate` | Report-only duplicate/stale/quality detection |
| 7 | `qdrant_learning_store` | Store procedural lessons (gated/manual) |
| 8 | `qdrant_learning_search` | Search procedural learnings |
| 9 | `qdrant_learning_preview` | Preview pending learning candidates |
| 10 | `qdrant_learning_approve` | Approve and store a candidate |

## 🧱 Lego Architecture

```
plugin/
├── __init__.py       ( 25)  # entry — import + register()
├── cli.py            (222)  # CLI subcommands (status, check-version, update)
├── config.py         ( 44)  # env var loading + constants
├── embeddings.py     ( 30)  # OpenAI-compatible embedding client
├── store.py          (181)  # QdrantStore — single-collection CRUD
├── schemas.py        (198)  # all 10 tool JSON schemas
├── provider.py       (431)  # QdrantMemoryProvider — wires everything
├── indexer.py        (359)  # FileIndexer — .md/.txt + manifest sync
├── learning.py       (258)  # LearningStore — procedural lessons
└── consolidation.py  (337)  # ConsolidationEngine — report-only
```

**Total: ~2,000 lines across 10 modules. Each file self-contained, independently testable. Swap any piece without touching the rest.**

## ⚠️ Safety Rules

| Rule | Enforcement |
|------|------------|
| Never delete any Qdrant collection | Zero `delete_collection()` calls in codebase |
| Each agent = own collection | Hard-scoped to `self._collection` at init |
| Dry-run first | `qdrant_forget`, `qdrant_index`, `qdrant_learning_approve` default dry_run=true |
| Consolidation = read-only | `qdrant_consolidate` finds issues, NEVER mutates |
| Learning = gated/manual | Auto-extraction disabled; all learnings explicitly stored |

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

# Or manual deploy
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/
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
# Expected: 10 tools listed
```

## 📄 Env Vars

| Variable | Required | Default |
|----------|----------|---------|
| `QDRANT_URL` | No | `http://localhost:6333` |
| `QDRANT_API_KEY` | No | — |
| `QDRANT_COLLECTION` | No | auto: `hermes_memories_<hostname>_<profile>` |
| `EMBEDDING_BASE_URL` | **Yes** | — |
| `EMBEDDING_API_KEY` | **Yes** | — |
| `EMBEDDING_MODEL` | No | `doubao-embedding-vision` |

## 📁 Full Repo

```
hermes-qdrant-integration/
├── README.md          # ← this file
├── SKILL.md           # Full setup guide + troubleshooting
├── PLAN.md            # Implementation plan
├── setup.sh           # Installer with --update/--force flags
└── plugin/
    ├── plugin.yaml    # Hermes plugin metadata (v0.2.0)
    ├── VERSION        # Plaintext version (v0.2.0)
    ├── cli.py         # CLI subcommands
    ├── __init__.py    # Entry point
    ├── config.py      # Config loading
    ├── embeddings.py  # Embedding client
    ├── store.py       # Qdrant CRUD wrapper
    ├── schemas.py     # Tool definitions
    ├── provider.py    # MemoryProvider impl
    ├── indexer.py     # File indexing
    ├── learning.py    # Learning store
    └── consolidation.py  # Memory consolidation
```

## 📜 License

MIT — use it, modify it, ship it.

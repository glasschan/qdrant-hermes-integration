---
name: hermes-qdrant-integration
description: "Plug-and-play Qdrant vector memory plugin for Hermes Agent — 6 tools, 9 modules, CLI commands, update mechanism. Install on any Hermes instance in 1 minute."
version: 2.8.0
author: Glass Chan + Paul
license: MIT
metadata:
  hermes:
    tags: [qdrant, memory, plugin, vector-db, hermes-agent, lego-architecture]
    related_skills: [hermes-memory-qdrant]
---

# Hermes Qdrant Integration Plugin

One-minute Qdrant vector memory for any Hermes Agent instance. Semantic long-term memory via local Qdrant + any OpenAI-compatible embedding API. **6 tools, 9 modules, 3 CLI commands.**

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
Each Hermes instance (hostname + profile combo) automatically gets its own namespace.

## Prerequisites

- Qdrant server running (local Docker: `docker run -p 6333:6333 qdrant/qdrant`, or self-hosted / Qdrant Cloud)
- An OpenAI-compatible embedding API endpoint + API key
- Hermes Agent installed (`hermes` CLI available)

## Quick Install (Recommended)

```bash
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash
```

The setup script handles **everything**:
- Installs to the standard user-installed plugin path (`~/.hermes/plugins/hermes-memory-qdrant/`)
- Installs `qdrant-client` in the Hermes venv
- Sets `memory.provider` in config
- Adds to `plugins.enabled`
- Prompts for env vars (URL, API keys, embedding model)

## Manual Install

```bash
# 1. Copy plugin to user-installed plugins dir
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/

# 2. Install dep
source ~/.hermes/hermes-agent/venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install qdrant-client

# 3. Configure
hermes config set memory.provider hermes-memory-qdrant

# 4. Enable plugin (for tool schemas)
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
QDRANT_URL=http://localhost:6333
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=your-embedding-model
EOF

# 6. Restart
hermes gateway restart
```

## Updating

```bash
# Check + upgrade
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --update

# Force reinstall
curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --force

# Via CLI (only available after install)
hermes hermes-memory-qdrant update
```

## Tools (6 total)

| Tool | Category | Description |
|------|----------|-------------|
| `qdrant_profile` | Core | Get all stored memories |
| `qdrant_search` | Core | Semantic search by meaning. Optional `recency_weight` (0.0-1.0) to favor fresh results. |
| `qdrant_remember` | Core | Store a fact (preference/fact/decision/goal/instruction). Optional `tags` array for filtering. |
| `qdrant_forget` | Core | Delete by point ID (dry-run first — safe default) |
| `qdrant_index` | Indexing | Index .md/.txt files/directories with manifest sync |
| `qdrant_consolidate` | Consolidation | Report-only duplicate/stale/quality detection |

> **v0.3.0 change:** Learning store removed (4 tool schemas dropped). See `references/learning-store-removal.md`.

## CLI Commands (v0.2.0+)

```bash
hermes hermes-memory-qdrant status            # Show plugin status + env vars
hermes hermes-memory-qdrant version           # Current vs latest release
hermes hermes-memory-qdrant update            # Upgrade to latest from GitHub
```

## Env Vars Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | ✓* | http://localhost:6333 | Qdrant server URL |
| `QDRANT_API_KEY` | — | — | Qdrant API key |
| `QDRANT_COLLECTION` | — | auto-generated† | Per-agent namespace |
| `EMBEDDING_BASE_URL` | ✓ | — | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_API_KEY` | ✓ | — | Embedding API key |
| `EMBEDDING_MODEL` | — | doubao-embedding-vision | Embedding model name |
| `QDRANT_DEDUP_THRESHOLD` | — | 0.85 | Cosine similarity for dedup |
| `QDRANT_DEDUP_ENABLED` | — | true | Enable pre-save dedup |
| `QDRANT_AUTO_SYNC` | — | false | Auto-save conversations to memory |
| `QDRANT_RECENCY_WEIGHT` | — | 0.0 | Recency bias in search (0.0-1.0) |

*> ✓ means recommended, URL defaults to localhost for local dev*
*† Auto-generated as `hermes_memories_<hostname>_<profile>`*

## Plugin Architecture

```
~/.hermes/plugins/hermes-memory-qdrant/   ← User path (tools + CLI)
~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant/  ← Bundled memory path (provider loading)

plugin/
├── __init__.py       (25 lines)  — entry point + register()
├── config.py         (56 lines)  — env var loading + constants + memory hygiene settings
├── embeddings.py     (30 lines)  — OpenAI-compatible embedding client
├── store.py         (~370 lines) — QdrantStore (single-collection CRUD + pre-save dedup + recency search)
├── schemas.py       (~147 lines) — all 6 tool schemas
├── provider.py      (~391 lines) — QdrantMemoryProvider (wires everything + memory hygiene)
├── indexer.py       (359 lines)  — FileIndexer (directory indexing)
├── consolidation.py (337 lines)  — ConsolidationEngine (report-only)
├── cli.py           (220 lines)  — CLI subcommands (status, version, update)
├── VERSION          (1 line)     — plaintext version string
├── plugin.yaml      (7 lines)    — Hermes plugin metadata (name, version, hooks, pip deps)
```

**Total: ~1,900 lines across 10 files.**

## Self-Healing: Hermes Namespace Bug Fix

The plugin **automatically** works around a known Hermes bug where user-installed memory provider plugins fail with `ModuleNotFoundError: No module named '_hermes_user_memory'`.

**Root cause:** Hermes loads user-installed memory providers under the `_hermes_user_memory.<name>` namespace but never registers `_hermes_user_memory` as a Python package in `sys.modules`. This breaks relative imports (`from .schemas import ...`) because Python can't resolve the parent package chain.

**Self-healing in `__init__.py`:** At load time, the plugin:
1. Detects the `_hermes_user_memory` namespace from its own `__name__`
2. Registers the missing parent package in `sys.modules`
3. Strips half-loaded submodules and reloads everything via `importlib`
4. Result: all relative imports resolve correctly

**No configuration, no dual-path install, no Hermes patches needed.** Single user-installed path works everywhere.

## Troubleshooting

| Problem | Likely Fix |
|---------|------------|
| `hermes doctor` shows "plugin not found" | Run setup.sh or check the plugin is at `~/.hermes/plugins/hermes-memory-qdrant/`. Run `hermes doctor --fix \| grep Memory` |
| `ModuleNotFoundError: No module named 'qdrant_client'` | Activate Hermes venv and run `pip install qdrant-client`. The venv is at `~/.hermes/hermes-agent/venv/` |
| `No module named '_hermes_user_memory'` | **Fixed in v0.4.0** — the plugin self-heals this. Run setup.sh --force to update. |
| Tools not showing up in session | Check `plugins.enabled` in `config.yaml` — must include `hermes-memory-qdrant`. Also check `memory.provider` is set. `hermes doctor` shows both. |
| Qdrant tools visible but not working | Run `hermes gateway restart` to reload the memory provider in running sessions |
| Memory not prefetching | Start a new session or run `/reset` |
| Embedding API returns 401 | Check `EMBEDDING_API_KEY` in `~/.hermes/.env` |
| `UserWarning: Api key with insecure connection` | Qdrant runs on HTTP — safe for local dev, add `PYTHONWARNINGS=ignore` to `.env` |

## Known Pitfalls

### 1. `hermes config set` serialises YAML lists as JSON strings

When you run `hermes config set plugins.enabled '[...]'`, it writes the value as a JSON string rather than a proper YAML list. Always verify the config and use the Python workaround (shown in manual install step 5) if needed.

### 2. Backup directory in `~/.hermes/plugins/` overwrites plugin version

If `hermes plugins list` shows a wrong version, check for `.bak.` directories in `~/.hermes/plugins/`. The plugin scanner walks directory names alphabetically and dedupes by name — a backup directory can overwrite the real version. Store backups outside `~/.hermes/plugins/`.

### 3. `_hermes_user_memory` namespace bug (Hermes agent)

See "Why Dual Install Paths?" above. This is a Hermes agent bug — fixed by installing to the bundled memory path. If Hermes ships a fix in a future version, the user-path-only install will work.

### 4. `setup.sh` needs pip installed in the venv

The Hermes venv may not have `pip`. The setup script runs `python3 -m ensurepip --upgrade` first. Manual installers should do the same.

### 5. `--update` requires GitHub access

The `--update` command downloads from GitHub. If the repo is private or the user has no internet, use `git clone` + manual copy instead.

## Repo

`/home/glasschan/paul-hermes/hermes-qdrant-integration`

Branch: `main` (v0.4.1)

## Verification

```bash
# Check provider status
hermes doctor --fix | grep "Memory Provider"

# Inside a session:
hermes chat -q "list all Qdrant tools you have access to"

# Should show: qdrant_profile, qdrant_search, qdrant_remember, qdrant_forget, qdrant_index, qdrant_consolidate
```
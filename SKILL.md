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
- Installs to **both** required paths (user path for tools/CLI + bundled memory path for memory provider discovery)
- Installs `qdrant-client` in the Hermes venv
- Sets `memory.provider` in config
- Adds to `plugins.enabled`
- Prompts for env vars (URL, API keys, embedding model)

## Manual Install

> **Important:** You MUST install to **two paths** for everything to work. This is due to how Hermes discovers memory providers — the `_load_provider_from_dir` function uses different module namespaces for bundled vs user paths, and only the bundled `plugins.memory` namespace properly supports relative imports.

```bash
# 1. Install to user path — required for plugin tool registration + CLI
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/

# 2. Install to bundled memory path — required for load_memory_provider() 
#    Uses plugins.memory.hermes-memory-qdrant namespace (relative imports work)
cp -r plugin/ ~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant/

# 3. Install qdrant-client in Hermes venv
source ~/.hermes/hermes-agent/venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install qdrant-client

# 4. Set as active memory provider
hermes config set memory.provider hermes-memory-qdrant

# 5. Enable plugin (for tool schemas to be available)
#    Note: use Python to avoid the JSON-string YAML bug in hermes config set
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
        print('✅ Added to plugins.enabled')
else:
    content += '\nplugins:\n  enabled:\n  - hermes-memory-qdrant\n'
    with open(path, 'w') as f: f.write(content)
"

# 6. Add to ~/.hermes/.env
echo 'QDRANT_URL=http://localhost:6333' >> ~/.hermes/.env
echo 'EMBEDDING_BASE_URL=https://your-api.com/v1' >> ~/.hermes/.env
echo 'EMBEDDING_API_KEY=sk-...' >> ~/.hermes/.env
echo 'EMBEDDING_MODEL=doubao-embedding-vision' >> ~/.hermes/.env

# 7. Restart gateway (if running)
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

## Why Dual Install Paths?

Hermes has **two separate discovery mechanisms** for memory providers:

| Mechanism | Scans | Used By |
|-----------|-------|---------|
| **Plugin system** (`plugins.enabled`) | `~/.hermes/plugins/<name>/` | Tool registration, CLI, `hermes plugins list` |
| **Memory provider loader** (`load_memory_provider`) | `plugins/memory/<name>/` (bundled) then `$HERMES_HOME/plugins/<name>/` (user) | `hermes doctor`, session runtime |

The user-installed path (`$HERMES_HOME/plugins/<name>/`) has a **known bug** in Hermes: when loading user-installed memory providers, the function `_load_provider_from_dir()` uses the namespace `_hermes_user_memory.<name>` but **never registers `_hermes_user_memory` as a package** in `sys.modules`. This causes relative imports (e.g., `from .schemas import ...`) to fail with `ModuleNotFoundError: No module named '_hermes_user_memory'`.

**Workaround:** Install to the **bundled memory path** (`plugins/memory/<name>/`) where the `plugins.memory` namespace is already properly set up. The bundled path takes precedence in `find_provider_dir()`, so the memory provider is discovered via the working namespace.

## Troubleshooting

| Problem | Likely Fix |
|---------|------------|
| `hermes doctor` shows "plugin not found" | The plugin must be installed at BOTH `~/.hermes/plugins/hermes-memory-qdrant/` AND `~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant/`. Run setup.sh or check manual install steps. |
| `ModuleNotFoundError: No module named 'qdrant_client'` | Activate Hermes venv and run `pip install qdrant-client`. The venv is at `~/.hermes/hermes-agent/venv/` |
| `No module named '_hermes_user_memory'` | The bundled memory path is missing or incomplete. Copy the full plugin to `plugins/memory/hermes-memory-qdrant/` with `__init__.py` |
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

Branch: `main` (v0.4.0)

## Verification

```bash
# Check provider status
hermes doctor --fix | grep "Memory Provider"

# Inside a session:
hermes chat -q "list all Qdrant tools you have access to"

# Should show: qdrant_profile, qdrant_search, qdrant_remember, qdrant_forget, qdrant_index, qdrant_consolidate
```
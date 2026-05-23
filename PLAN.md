# Qdrant Integration Improvement Plan

## v0.2.0 — Plugin Alignment & Update Mechanism

Branch: `feature/plugin-align-update-mechanism`

### Changes

1. **Install path moved** — from bundled (`~/.hermes/hermes-agent/plugins/memory/`) to user-installed (`~/.hermes/plugins/`), aligning with official Hermes direction
2. **Update mechanism** — `setup.sh --update` / `setup.sh --force` with semver comparison + backup
3. **CLI subcommands** — `hermes memory-qdrant {status,version,update}` via new `plugin/cli.py`
4. **Version tracking** — `plugin/VERSION` plaintext file + bumped `plugin.yaml` to 0.2.0
5. **Old bundled cleanup** — setup.sh automatically removes old bundled copy

### Files Changed
- `setup.sh` — new path, --update/--force flags, version check, backup, old cleanup
- `README.md` — new install paths, added Updating section
- `SKILL.md` — new paths, full Lego architecture, update instructions
- `plugin/cli.py` — new file, CLI commands for status/version/update
- `plugin/VERSION` — new file, plaintext version
- `plugin/plugin.yaml` — bumped 0.1.0 → 0.2.0

### Verification
```bash
# Deploy to user path
cp -r plugin/ ~/.hermes/plugins/hermes-memory-qdrant/

# Verify
hermes doctor | grep memory
hermes memory-qdrant version
hermes memory-qdrant status

# Functional test
hermes chat -q "list all Qdrant tools you have access to"
```
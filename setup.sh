#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║  Hermes Qdrant Memory — Lego Plugin v0.1  ║"
echo "║  10 tools · 9 modules · 1 command       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PLUGIN_DIR="$HOME/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant"

# 1. Backup existing plugin if present
if [ -f "$PLUGIN_DIR/__init__.py" ]; then
    BACKUP_DIR="$PLUGIN_DIR.bak.$(date +%Y%m%d_%H%M%S)"
    cp -r "$PLUGIN_DIR" "$BACKUP_DIR"
    echo "✅ Backup: $BACKUP_DIR"
fi

# 2. Copy all Lego modules
mkdir -p "$PLUGIN_DIR"
cp plugin/*.py "$PLUGIN_DIR/"
cp plugin/plugin.yaml "$PLUGIN_DIR/"
echo "✅ 9 modules + plugin.yaml deployed to $PLUGIN_DIR"

# 3. Install qdrant-client
HERMES_DIR="$HOME/.hermes/hermes-agent"
if [ -d "$HERMES_DIR/venv" ]; then
    VIRTUAL_ENV="$HERMES_DIR/venv" uv pip install qdrant-client 2>/dev/null || \
    "$HERMES_DIR/venv/bin/pip" install qdrant-client 2>/dev/null || \
    pip3 install qdrant-client
elif [ -d "$HERMES_DIR/.venv" ]; then
    VIRTUAL_ENV="$HERMES_DIR/.venv" uv pip install qdrant-client 2>/dev/null || \
    "$HERMES_DIR/.venv/bin/pip" install qdrant-client 2>/dev/null || \
    pip3 install qdrant-client
else
    pip3 install qdrant-client 2>/dev/null || pip install qdrant-client
fi
echo "✅ qdrant-client installed"

# 4. Set memory provider
hermes config set memory.provider hermes-memory-qdrant
echo "✅ memory.provider = hermes-memory-qdrant"

# 5. Interactive env var setup
echo ""
echo "─── Environment Variables ───"
echo "(press Enter to skip optional fields)"
echo ""

read -rp "QDRANT_URL [http://localhost:6333]: " qurl
read -rp "QDRANT_API_KEY (optional): " qkey
echo ""
echo "QDRANT_COLLECTION — MUST be unique per deployment!"
echo "Leave empty to auto-generate: hermes_memories_<hostname>_<profile>"
read -rp "QDRANT_COLLECTION: " qcoll
echo ""
read -rp "EMBEDDING_BASE_URL [required]: " eurl
read -rp "EMBEDDING_API_KEY [required]: " ekey
read -rp "EMBEDDING_MODEL [doubao-embedding-vision]: " emodel

cat >> "$HOME/.hermes/.env" << EOF
${qurl:+QDRANT_URL=$qurl}
${qkey:+QDRANT_API_KEY=$qkey}
${qcoll:+QDRANT_COLLECTION=$qcoll}
${eurl:+EMBEDDING_BASE_URL=$eurl}
${ekey:+EMBEDDING_API_KEY=$ekey}
${emodel:+EMBEDDING_MODEL=$emodel}
EOF
echo "✅ Env vars written to ~/.hermes/.env"

# 6. Verify
echo ""
echo "─── Verification ───"
hermes doctor --fix 2>&1 | grep -E "(Memory Provider|hermes-memory-qdrant)" || true

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  🎉 DONE!                                ║"
echo "║                                          ║"
echo "║  Test: hermes chat -q \"list all Qdrant  ║"
echo "║         tools you have access to\"        ║"
echo "╚══════════════════════════════════════════╝"

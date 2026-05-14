#!/usr/bin/env bash
set -euo pipefail

echo "=== Hermes Qdrant Memory — Setup ==="

# 1. Copy plugin
PLUGIN_DIR="$HOME/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant"
mkdir -p "$PLUGIN_DIR"
cp plugin/__init__.py "$PLUGIN_DIR/"
cp plugin/plugin.yaml "$PLUGIN_DIR/"
echo "✅ Plugin copied to $PLUGIN_DIR"

# 2. Install qdrant-client
cd "$HOME/.hermes/hermes-agent"
if command -v uv &>/dev/null; then
    VIRTUAL_ENV="$HOME/.hermes/hermes-agent/venv" uv pip install qdrant-client
elif command -v pip3 &>/dev/null; then
    pip3 install qdrant-client
else
    pip install qdrant-client
fi
echo "✅ qdrant-client installed"

# 3. Set config
hermes config set memory.provider hermes-memory-qdrant
echo "✅ memory.provider set"

# 4. Prompt for env vars
echo ""
echo "=== 請填入以下 Env Vars（留空跳過）==="
read -rp "QDRANT_URL [http://localhost:6333]: " qurl
read -rp "QDRANT_API_KEY: " qkey
echo ""
echo "QDRANT_COLLECTION (留空 = auto-generate hermes_memories_<hostname>_<profile>)"
echo "⚠️  每部機 / 每個 profile 要唔同名！Plugin 唔會掂其他 collection！"
read -rp "QDRANT_COLLECTION: " qcoll
read -rp "EMBEDDING_BASE_URL: " eurl
read -rp "EMBEDDING_API_KEY: " ekey
read -rp "EMBEDDING_MODEL [doubao-embedding-vision]: " emodel

cat >> "$HOME/.hermes/.env" << EOF
${qurl:+QDRANT_URL=$qurl}
${qkey:+QDRANT_API_KEY=$qkey}
${qcoll:+QDRANT_COLLECTION=$qcoll}
${eurl:+EMBEDDING_BASE_URL=$eurl}
${ekey:+EMBEDDING_API_KEY=$ekey}
${emodel:+EMBEDDING_MODEL=$emodel}
EOF
echo "✅ Env vars appended to ~/.hermes/.env"

# 5. Verify
echo ""
echo "=== Verification ==="
hermes doctor --fix
echo ""
echo "🎉 Done! Run 'hermes chat -q \"qdrant_remember 記低：Hello Qdrant\"' to test."

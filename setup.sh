#!/usr/bin/env bash
set -euo pipefail

# ── Constants ─────────────────────────────────────────────────────────────
VERSION="0.2.0"
REPO="glasschan/qdrant-hermes-integration"
PLUGIN_NAME="hermes-memory-qdrant"
PLUGIN_DIR="$HOME/.hermes/plugins/$PLUGIN_NAME"

# ── Flags ─────────────────────────────────────────────────────────────────
MODE="install"  # install | update | force
while [[ $# -gt 0 ]]; do
    case "$1" in
        --update) MODE="update"; shift ;;
        --force)  MODE="force";  shift ;;
        --help|-h)
            echo "Usage: bash setup.sh [--update|--force]"
            echo ""
            echo "  (no flag)  Fresh install"
            echo "  --update   Check for update and upgrade if newer"
            echo "  --force    Force reinstall regardless of version"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash setup.sh [--update|--force]"
            exit 1
            ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Hermes Qdrant Memory Plugin v$VERSION       ║"
echo "║  10 tools · 9 modules · 1 command       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Version check (for update mode) ──────────────────────────────────────
if [[ "$MODE" == "update" || "$MODE" == "force" ]]; then
    if [ ! -d "$PLUGIN_DIR" ]; then
        echo "⚠️  Plugin not installed at $PLUGIN_DIR"
        echo "   Run without --update for a fresh install."
        exit 1
    fi

    CURRENT_VERSION=""
    VERSION_FILE="$PLUGIN_DIR/VERSION"
    if [ -f "$VERSION_FILE" ]; then
        CURRENT_VERSION=$(cat "$VERSION_FILE" | tr -d 'vV \n\t')
    fi

    if [[ -z "$CURRENT_VERSION" && -f "$PLUGIN_DIR/plugin.yaml" ]]; then
        CURRENT_VERSION=$(grep '^version:' "$PLUGIN_DIR/plugin.yaml" | sed 's/.*: *//' | tr -d 'vV \n\t')
    fi

    echo "   Current: v${CURRENT_VERSION:-unknown}"
    echo "   Latest:  v$VERSION"
    echo ""

    if [[ "$MODE" == "update" ]]; then
        if [ "$(printf '%s\n' "$CURRENT_VERSION" "$VERSION" | sort -V | tail -1)" == "$CURRENT_VERSION" ] && [ "$CURRENT_VERSION" != "$VERSION" ]; then
            echo "   ✅ Already up to date (v$CURRENT_VERSION)"
            echo "   Use --force to reinstall."
            exit 0
        fi
        echo "   🚀 Upgrading v$CURRENT_VERSION → v$VERSION ..."
    else
        echo "   🔄 Force reinstall (v${CURRENT_VERSION:-unknown} → v$VERSION)"
    fi

    # Backup current plugin
    BACKUP_DIR="$PLUGIN_DIR.bak.v$CURRENT_VERSION"
    if [ ! -d "$BACKUP_DIR" ]; then
        cp -r "$PLUGIN_DIR" "$BACKUP_DIR"
        echo "   ✅ Backup: $BACKUP_DIR"
    else
        echo "   ⏭️  Backup already exists: $BACKUP_DIR"
    fi
fi

# ── Fresh install ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR"

# If running via curl | bash, we need the plugin/ subtree; it's at SCRIPT_DIR/plugin/
if [ ! -d "$SOURCE_DIR/plugin" ]; then
    # Try the repo root
    SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [ ! -d "$SOURCE_DIR/plugin" ]; then
    echo "❌ Can't find plugin/ directory. Run this script from the repo root."
    exit 1
fi

# 1. Create user plugin directory
mkdir -p "$PLUGIN_DIR"
echo "✅ Plugin directory: $PLUGIN_DIR"

# 2. Copy all Lego modules + metadata
cp "$SOURCE_DIR/plugin/"*.py "$PLUGIN_DIR/"
cp "$SOURCE_DIR/plugin/plugin.yaml" "$PLUGIN_DIR/"
cp "$SOURCE_DIR/plugin/VERSION" "$PLUGIN_DIR/" 2>/dev/null || true
echo "✅ Modules + metadata deployed"

# 3. Clean up old bundled copy (if exists)
OLD_BUNDLED="$HOME/.hermes/hermes-agent/plugins/memory/$PLUGIN_NAME"
if [ -d "$OLD_BUNDLED" ]; then
    cp -r "$OLD_BUNDLED" "$OLD_BUNDLED.bak.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
    rm -rf "$OLD_BUNDLED"
    echo "✅ Removed old bundled copy: $OLD_BUNDLED"
fi

# 4. Install qdrant-client
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

# 5. Set memory provider
hermes config set memory.provider "$PLUGIN_NAME" 2>/dev/null || true
echo "✅ memory.provider = $PLUGIN_NAME"

# 6. Interactive env var setup (fresh install only, skip on update)
if [[ "$MODE" != "update" && "$MODE" != "force" ]]; then
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
fi

# 7. Write VERSION file (in case it wasn't copied)
echo "v$VERSION" > "$PLUGIN_DIR/VERSION"

# 8. Verify
echo ""
echo "─── Verification ───"
hermes doctor --fix 2>&1 | grep -E "(Memory Provider|$PLUGIN_NAME)" || true

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  🎉 DONE! v$VERSION                      ║"
echo "║                                          ║"
echo "║  Installed at:                           ║"
echo "║    $PLUGIN_DIR        ║"
echo "║                                          ║"
echo "║  Test: hermes chat -q \"list all Qdrant   ║"
echo "║         tools you have access to\"         ║"
echo "╚══════════════════════════════════════════╝"
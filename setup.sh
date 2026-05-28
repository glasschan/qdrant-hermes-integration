#!/usr/bin/env bash
set -euo pipefail

# ── Constants ─────────────────────────────────────────────────────────────
VERSION="0.4.0"
REPO="glasschan/qdrant-hermes-integration"
PLUGIN_NAME="hermes-memory-qdrant"
USER_PLUGIN_DIR="$HOME/.hermes/plugins/$PLUGIN_NAME"
BACKUP_DIR="$HOME/.hermes/plugin-backups"

# ── Flags ─────────────────────────────────────────────────────────────────
MODE="install"
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
echo "║  6 tools · 9 modules · 3 CLI commands   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Version check (update/force mode) ────────────────────────────────────
if [[ "$MODE" == "update" || "$MODE" == "force" ]]; then
    if [ ! -d "$USER_PLUGIN_DIR" ]; then
        echo "⚠️  Plugin not installed at $USER_PLUGIN_DIR"
        echo "   Run without --update for a fresh install."
        exit 1
    fi

    CURRENT_VERSION=""
    VERSION_FILE="$USER_PLUGIN_DIR/VERSION"
    [ -f "$VERSION_FILE" ] && CURRENT_VERSION=$(cat "$VERSION_FILE" | tr -d 'vV \\n\\t')
    if [[ -z "$CURRENT_VERSION" && -f "$USER_PLUGIN_DIR/plugin.yaml" ]]; then
        CURRENT_VERSION=$(grep '^version:' "$USER_PLUGIN_DIR/plugin.yaml" | sed 's/.*: *//' | tr -d 'vV \\n\\t')
    fi

    echo "   Current: v${CURRENT_VERSION:-unknown}"
    echo "   Latest:  v$VERSION"
    echo ""

    if [[ "$MODE" == "update" ]]; then
        if [ "$CURRENT_VERSION" == "$VERSION" ]; then
            echo "   ✅ Already up to date (v$CURRENT_VERSION)"
            echo "   Use --force to reinstall."
            exit 0
        fi
        NEWER=$(printf '%s\\n' "$CURRENT_VERSION" "$VERSION" | sort -V | tail -1)
        if [ "$NEWER" == "$CURRENT_VERSION" ]; then
            echo "   ✅ Already up to date (v$CURRENT_VERSION, ahead of v$VERSION)"
            echo "   Use --force to reinstall."
            exit 0
        fi
        echo "   🚀 Upgrading v$CURRENT_VERSION → v$VERSION ..."
    else
        echo "   🔄 Force reinstall (v${CURRENT_VERSION:-unknown} → v$VERSION)"
    fi

    # Backup to plugin-backups/ (outside plugins/ dir to avoid version hijack)
    TS=$(date +%Y%m%d_%H%M%S)
    mkdir -p "$BACKUP_DIR"
    BAK="$BACKUP_DIR/$PLUGIN_NAME.v$CURRENT_VERSION.$TS"
    if [ ! -d "$BAK" ]; then
        cp -r "$USER_PLUGIN_DIR" "$BAK"
        echo "   ✅ Backup: $BAK"
    fi
fi

# ── Resolve source directory ─────────────────────────────────────────────
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null || true)"
fi
SOURCE_DIR="$SCRIPT_DIR"

if [ -z "$SCRIPT_DIR" ] || [ ! -d "$SOURCE_DIR/plugin" ]; then
    if [ -d "$(pwd)/plugin" ]; then
        SOURCE_DIR="$(pwd)"
    elif command -v curl &>/dev/null; then
        echo "📥 Downloading plugin files from GitHub..."
        TMP_DIR=$(mktemp -d)
        curl -sL "https://api.github.com/repos/$REPO/tarball/main" | tar xz -C "$TMP_DIR" --strip=1 2>/dev/null || {
            echo "❌ Failed to download from GitHub."
            rm -rf "$TMP_DIR"
            exit 1
        }
        SOURCE_DIR="$TMP_DIR"
    else
        echo "❌ Can't find plugin/ directory. Run this script from the cloned repo."
        exit 1
    fi
fi

# ── 1. Install plugin files to user path ─────────────────────────────────
echo ""
echo "─── Installing plugin files ───"
mkdir -p "$USER_PLUGIN_DIR"
cp "$SOURCE_DIR/plugin/"*.py "$USER_PLUGIN_DIR/"
cp "$SOURCE_DIR/plugin/plugin.yaml" "$USER_PLUGIN_DIR/"
cp "$SOURCE_DIR/plugin/VERSION" "$USER_PLUGIN_DIR/" 2>/dev/null || true
echo "   ✅ $USER_PLUGIN_DIR"

# ── 2. Install qdrant-client in Hermes venv ──────────────────────────────
echo ""
echo "─── Installing Python dependencies ───"
HERMES_VENV=""
for v in "$HOME/.hermes/hermes-agent/venv" "$HOME/.hermes/hermes-agent/.venv"; do
    [ -f "$v/bin/python3" ] && HERMES_VENV="$v" && break
done

if [ -n "$HERMES_VENV" ]; then
    echo "   Hermes venv: $HERMES_VENV"
    "$HERMES_VENV/bin/python3" -m ensurepip --upgrade 2>/dev/null || true
    "$HERMES_VENV/bin/python3" -m pip install qdrant-client 2>&1 | tail -2
    echo "✅ qdrant-client installed"
else
    echo "⚠️  Hermes venv not found. Trying system-wide pip..."
    pip3 install qdrant-client 2>/dev/null || pip install qdrant-client || \
        echo "⚠️  Install manually: pip install qdrant-client"
fi

# ── 3. Set memory.provider ───────────────────────────────────────────────
echo ""
echo "─── Configuring Hermes ───"
hermes config set memory.provider "$PLUGIN_NAME" 2>/dev/null || {
    CONFIG="$HOME/.hermes/config.yaml"
    if grep -q "provider:" "$CONFIG"; then
        sed -i "s/provider: .*/provider: $PLUGIN_NAME/" "$CONFIG"
    else
        echo "   ⚠️  Could not set memory.provider. Set manually:"
        echo "      hermes config set memory.provider $PLUGIN_NAME"
    fi
}
echo "✅ memory.provider = $PLUGIN_NAME"

# ── 4. Add to plugins.enabled ────────────────────────────────────────────
echo ""
echo "─── Enabling plugin ───"
if hermes plugins list 2>/dev/null | grep -q "$PLUGIN_NAME.*enabled"; then
    echo "✅ Already enabled in plugins.enabled"
else
    python3 -c "
import re
path = '$HOME/.hermes/config.yaml'
with open(path) as f: content = f.read()
match = re.search(r'enabled:\s*\n(\s+- .+\n?)*', content)
if match:
    block = match.group()
    if '$PLUGIN_NAME' not in block:
        content = content.replace(block, block.rstrip() + '\n  - $PLUGIN_NAME\n')
        with open(path, 'w') as f: f.write(content)
        print('✅ Added to plugins.enabled')
    else:
        print('✅ Already in plugins.enabled')
else:
    content += '\nplugins:\n  enabled:\n  - $PLUGIN_NAME\n'
    with open(path, 'w') as f: f.write(content)
    print('✅ Created plugins.enabled section')
" 2>&1 || echo "   ⚠️  Could not update plugins.enabled. Add manually:"
fi

# ── 5. Interactive env vars (fresh install only) ─────────────────────────
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

    {
        echo ""
        echo "# Qdrant Memory"
        echo "QDRANT_URL=${qurl:-http://localhost:6333}"
        [ -n "$qkey" ]   && echo "QDRANT_API_KEY=$qkey"
        [ -n "$qcoll" ]  && echo "QDRANT_COLLECTION=$qcoll"
        [ -n "$eurl" ]   && echo "EMBEDDING_BASE_URL=$eurl"
        [ -n "$ekey" ]   && echo "EMBEDDING_API_KEY=$ekey"
        [ -n "$emodel" ] && echo "EMBEDDING_MODEL=$emodel"
    } >> "$HOME/.hermes/.env"
    echo "✅ Env vars written to ~/.hermes/.env"
fi

# ── 6. Clean up temp ─────────────────────────────────────────────────────
if [ -n "${TMP_DIR:-}" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
fi

# ── 7. Write VERSION ─────────────────────────────────────────────────────
echo "v$VERSION" > "$USER_PLUGIN_DIR/VERSION"

# ── 8. Verify ────────────────────────────────────────────────────────────
echo ""
echo "─── Verification ───"
hermes doctor --fix 2>&1 | grep -E "(Memory Provider|$PLUGIN_NAME)" || true

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  🎉 DONE! v$VERSION                      ║"
echo "║                                          ║"
echo "║  Installed at:                           ║"
echo "║    $USER_PLUGIN_DIR   ║"
echo "║                                          ║"
echo "║  Test:                                   ║"
echo "║    hermes chat -q \"list all Qdrant       ║"
echo "║      tools you have access to\"            ║"
echo "╚══════════════════════════════════════════╝"
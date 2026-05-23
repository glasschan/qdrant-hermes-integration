"""CLI subcommands for hermes-memory-qdrant plugin.

Registered via register_cli(subparser) as per Hermes plugin CLI convention.
Loaded by discover_plugin_cli_commands() in plugins/memory/__init__.py.

Usage: hermes hermes-memory-qdrant [--version|--update|--status]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_plugin_dir() -> Path:
    """Return the plugin directory (where this file lives)."""
    return Path(__file__).parent.resolve()


def _read_version(path: Path) -> str:
    """Read version from VERSION file or plugin.yaml."""
    vfile = path / "VERSION"
    if vfile.exists():
        return vfile.read_text(encoding="utf-8").strip()
    yaml_file = path / "plugin.yaml"
    if yaml_file.exists():
        for line in yaml_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def _get_hermes_home() -> Path:
    """Resolve HERMES_HOME."""
    env = os.environ.get("HERMES_HOME", "")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _check_latest_version() -> str:
    """Fetch latest version tag from GitHub releases."""
    try:
        result = subprocess.run(
            [
                "curl", "-sL",
                "https://api.github.com/repos/glasschan/qdrant-hermes-integration/releases/latest",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("tag_name", "unknown")
    except Exception:
        pass
    return "unknown"


# ── Handlers ─────────────────────────────────────────────────────────────

def _do_status() -> None:
    """Show plugin status."""
    plugin_dir = _get_plugin_dir()
    version = _read_version(plugin_dir)
    hermes_home = _get_hermes_home()

    print(f"Plugin:         hermes-memory-qdrant")
    print(f"Version:        {version}")
    print(f"Path:           {plugin_dir}")
    print(f"Hermes Home:    {hermes_home}")

    py_files = list(plugin_dir.glob("*.py"))
    print(f"Modules:        {len(py_files)} Python files")
    print(f"Tools:          10 (qdrant_profile/search/remember/forget/index/consolidate/learning_*)")

    # Check env vars
    env_file = hermes_home / ".env"
    if env_file.exists():
        found = []
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if any(k in line for k in ("QDRANT_", "EMBEDDING_")):
                key = line.split("=", 1)[0]
                val = line.split("=", 1)[1] if "=" in line else ""
                if "KEY" in key or "SECRET" in key or "TOKEN" in key:
                    val = "***" if val else ""
                found.append(f"{key}={val}")
        if found:
            print(f"\nConfigured env vars:")
            for f in found:
                print(f"  {f}")
        else:
            print(f"\n⚠️  No Qdrant env vars found in {env_file}")


def _do_version() -> None:
    """Show current version and check latest available."""
    plugin_dir = _get_plugin_dir()
    current = _read_version(plugin_dir)
    latest = _check_latest_version()

    print(f"Current: {current}")
    print(f"Latest:  {latest}")

    if latest != "unknown" and current != "unknown":
        c = current.lstrip("vV")
        l = latest.lstrip("vV")
        if c == l:
            print("Status:  ✅ Up to date")
        else:
            try:
                result = subprocess.run(
                    ["bash", "-c", f"printf '%s\\n' '{c}' '{l}' | sort -V | tail -1"],
                    capture_output=True, text=True, timeout=5,
                )
                newer = result.stdout.strip()
                if newer == l:
                    print("Status:  🔄 Update available — run 'hermes hermes-memory-qdrant --update'")
                else:
                    print("Status:  ✅ Up to date (ahead of latest release)")
            except Exception:
                print("Status:  ⚠️  Could not compare versions")


def _do_update() -> None:
    """Run setup.sh --update to upgrade the plugin.

    Always downloads from GitHub to ensure we get the latest version.
    """
    print("Downloading latest setup.sh from GitHub...")
    try:
        subprocess.run(
            [
                "bash", "-c",
                "curl -sL https://raw.githubusercontent.com/glasschan/qdrant-hermes-integration/main/setup.sh | bash -s -- --update",
            ],
            check=True, timeout=120,
        )
    except subprocess.CalledProcessError:
        print("❌ Update failed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Update failed: {e}")
        sys.exit(1)


# ── Registration ─────────────────────────────────────────────────────────

def register_cli(subparser) -> None:
    """Configure the 'hermes hermes-memory-qdrant' subcommand.

    Uses flags (--version, --update, --status) instead of sub-subparsers
    for simplicity and compatibility with Hermes CLI dispatch.
    """
    subparser.add_argument(
        "--check-version", "-c",
        action="store_true",
        help="Show current version and check latest available",
    )
    subparser.add_argument(
        "--update", "-u",
        action="store_true",
        help="Check for update and upgrade",
    )
    subparser.add_argument(
        "--status",
        action="store_true",
        help="Show plugin status and configuration",
    )


# ── Top-level handler ────────────────────────────────────────────────────
# Hermes looks for a function named '<plugin-name>_command' in cli.py.
# For 'hermes-memory-qdrant', it's hermes_memory_qdrant_command (dash → underscore).

def hermes_memory_qdrant_command(args: argparse.Namespace) -> None:
    """Handle 'hermes hermes-memory-qdrant' command."""
    if args.check_version:
        _do_version()
    elif args.update:
        _do_update()
    elif args.status:
        _do_status()
    else:
        print("hermes-memory-qdrant plugin — Qdrant vector memory for Hermes Agent")
        print()
        print("Usage: hermes hermes-memory-qdrant [--check-version|--update|--status]")
        print()
        print("Examples:")
        print("  hermes hermes-memory-qdrant --status         Show plugin status")
        print("  hermes hermes-memory-qdrant --check-version  Check versions")
        print("  hermes hermes-memory-qdrant --update         Upgrade to latest")


# Hermes CLI lookup uses the raw provider name (with dashes) as the function
# name via getattr(). Since Python identifiers can't contain dashes, we
# alias it here so the discovery system finds the handler.
import sys as _sys
_caller = _sys.modules[__name__]
setattr(_caller, "hermes-memory-qdrant_command", hermes_memory_qdrant_command)
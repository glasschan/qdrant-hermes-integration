"""CLI subcommands for hermes-memory-qdrant plugin.

Registered via register_cli(subparser) as per Hermes plugin CLI convention.
Loaded by discover_plugin_cli_commands() in plugins/memory/__init__.py.

Usage: hermes hermes-memory-qdrant <status|version|update>
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


# ── Subcommand handlers ──────────────────────────────────────────────────

def _cmd_status() -> None:
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


def _cmd_version() -> None:
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
                    print("Status:  🔄 Update available — run 'hermes hermes-memory-qdrant update'")
                else:
                    print("Status:  ✅ Up to date (ahead of latest release)")
            except Exception:
                print("Status:  ⚠️  Could not compare versions")


def _cmd_update() -> None:
    """Run setup.sh --update to upgrade the plugin."""
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
    """Build the 'hermes hermes-memory-qdrant' argparse tree.

    Follows the official Hermes plugin CLI convention from:
    https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin
    """
    subs = subparser.add_subparsers(dest="memory_qdrant_command", title="subcommands")

    p = subs.add_parser("status", help="Show plugin status and configuration")
    p.set_defaults(handler=_cmd_status)

    p = subs.add_parser("version", help="Show current and latest available version")
    p.set_defaults(handler=_cmd_version)

    p = subs.add_parser("update", help="Check for update and upgrade")
    p.set_defaults(handler=_cmd_update)


# ── Dispatch ─────────────────────────────────────────────────────────────
# Hermes CLI looks for '<provider-name>_command' in cli.py via getattr().
# Since provider name uses dashes ("hermes-memory-qdrant") but Python
# identifiers can't, we alias it with setattr().

def hermes_memory_qdrant_command(args: argparse.Namespace) -> None:
    """Dispatch 'hermes hermes-memory-qdrant <subcommand>'."""
    handler = getattr(args, "handler", None)
    if handler:
        handler()
    else:
        print("Usage: hermes hermes-memory-qdrant <status|version|update>")
        print()
        print("Subcommands:")
        print("  status    Show plugin status and configuration")
        print("  version   Show current and latest available version")
        print("  update    Check for update and upgrade")


# Alias so discover_plugin_cli_commands() finds the handler
import sys as _sys
_caller = _sys.modules[__name__]
setattr(_caller, "hermes-memory-qdrant_command", hermes_memory_qdrant_command)
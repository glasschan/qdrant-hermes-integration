"""Shared test fixtures and mocks.

Mocks Hermes-only modules BEFORE any test module imports plugin.
This MUST happen at import time (not in a fixture) because
plugin/__init__.py has module-level side effects.
"""
import sys
import types
from unittest.mock import MagicMock
from pathlib import Path

# ── Mock all Hermes-only dependencies at import time ──────────────────
# plugin/__init__.py does QdrantMemoryProvider = _import_provider()
# which triggers provider.py → agent.memory_provider import.
# We must mock these BEFORE any test module imports from plugin.

# 1. agent.memory_provider.MemoryProvider
_mock_agent = types.ModuleType("agent")
_mock_agent_mp = types.ModuleType("agent.memory_provider")


class _MockMemoryProvider:
    """Minimal MemoryProvider stub for testing."""
    def __init__(self): pass
    name = "mock"
    def is_available(self): return True
    def initialize(self, *a, **kw): pass
    def shutdown(self): pass
    def system_prompt_block(self): return ""
    def get_tool_schemas(self): return []
    def handle_tool_call(self, *a, **kw): return ""
    def prefetch(self, *a, **kw): return ""
    def queue_prefetch(self, *a, **kw): pass
    def sync_turn(self, *a, **kw): pass


_mock_agent_mp.MemoryProvider = _MockMemoryProvider
_mock_agent.memory_provider = _mock_agent_mp
sys.modules.setdefault("agent", _mock_agent)
sys.modules.setdefault("agent.memory_provider", _mock_agent_mp)

# 2. tools.registry.tool_error
_mock_tools = types.ModuleType("tools")
_mock_registry = types.ModuleType("tools.registry")
_mock_registry.tool_error = lambda msg: f'{{"error": "{msg}"}}'
sys.modules.setdefault("tools", _mock_tools)
sys.modules.setdefault("tools.registry", _mock_registry)

# 3. hermes_constants.get_hermes_home
_mock_hc = types.ModuleType("hermes_constants")
_mock_hc.get_hermes_home = lambda: Path("/tmp/test_hermes_home")
sys.modules.setdefault("hermes_constants", _mock_hc)

# 4. hermes_cli.plugins.get_plugin_manager
_mock_cli = types.ModuleType("hermes_cli")
_mock_plugins = types.ModuleType("hermes_cli.plugins")
_mock_plugins.get_plugin_manager = lambda: MagicMock(_hooks={})
sys.modules.setdefault("hermes_cli", _mock_cli)
sys.modules.setdefault("hermes_cli.plugins", _mock_plugins)

"""Qdrant vector memory plugin — plug-and-play Hermes memory provider.

Modular architecture (Lego-style):
  config.py      — constants + env-var loading
  embeddings.py  — OpenAI-compatible embedding client
  store.py       — QdrantStore (single-collection CRUD wrapper)
  schemas.py     — all 6 tool schemas
  indexer.py     — FileIndexer (directory indexing + manifest sync)
  consolidation.py — ConsolidationEngine (report-only dedup/stale/quality)
  provider.py    — QdrantMemoryProvider (wires everything together)

Hermes compatibility:
  Installs to ~/.hermes/plugins/hermes-memory-qdrant/ (user-installed path).
  No bundled-path copy needed — this module self-heals the _hermes_user_memory
  namespace bug in Hermes's _load_provider_from_dir().
"""

from __future__ import annotations

import sys as _sys
import types as _types
from pathlib import Path as _Path


def _ensure_namespace() -> None:
    """Self-heal the _hermes_user_memory namespace bug in Hermes.

    Hermes loads user-installed memory provider plugins under the
    ``_hermes_user_memory.<name>`` namespace (via importlib), but never
    registers ``_hermes_user_memory`` as a package in ``sys.modules``.
    This breaks relative imports (``from .provider import ...``) because
    Python can't resolve the parent package chain.

    This function:
    1. Registers the parent namespace (e.g. ``_hermes_user_memory``) in
       ``sys.modules`` so Python can resolve relative import paths.
    2. Registers this plugin's own package with ``__path__`` so sibling
       submodule imports (``from .schemas import ...`` in provider.py)
       resolve correctly.
    3. Strips any half-loaded submodules from sys.modules so they get
       cleanly re-imported via importlib below.
    """
    parts = __name__.split(".")

    # 1. Register parent chain (e.g. _hermes_user_memory) if missing
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in _sys.modules:
            pkg = _types.ModuleType(parent)
            pkg.__path__ = []
            pkg.__package__ = parent
            _sys.modules[parent] = pkg

    # 2. Ensure this package has __path__ for submodule resolution
    pkg = _sys.modules[__name__]
    if not hasattr(pkg, "__path__") or not pkg.__path__:
        pkg.__path__ = [str(_Path(__file__).parent)]
        pkg.__package__ = __name__

    # 3. Strip all submodules so importlib re-does them cleanly now that
    #    the parent namespace is properly set up. importlib.import_module()
    #    will reload everything fresh.
    for mod_name in list(_sys.modules.keys()):
        if mod_name.startswith(f"{__name__}.") and mod_name != __name__:
            del _sys.modules[mod_name]


def _import_provider():
    """Import QdrantMemoryProvider via importlib (namespace-safe).

    Uses importlib.import_module() instead of a ``from .provider import``
    statement, so it works under any namespace (including the broken
    ``_hermes_user_memory`` namespace).
    """
    import importlib as _il

    _ensure_namespace()
    _mod = _il.import_module(f"{__name__}.provider")
    return _mod.QdrantMemoryProvider


QdrantMemoryProvider = _import_provider()


def register(ctx) -> None:
    """Register the Qdrant memory provider plugin.

    NOTE: ctx here is a _ProviderCollector (from plugins/memory/__init__.py)
    whose register_hook() is a no-op.  Hooks are registered separately in
    provider.py's _register_hooks_with_plugin_manager() during initialize(),
    which bypasses _ProviderCollector and writes to the real PluginManager's
    _hooks dict directly.
    """
    provider = QdrantMemoryProvider()
    ctx.register_memory_provider(provider)
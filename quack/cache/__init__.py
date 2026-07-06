# Copyright (c) 2025-2026, Tri Dao.
"""Persistent kernel-cache utilities for QuACK.

Public API
----------

Persistent ``.o`` cache:
* :func:`jit_cache` — decorator that wraps a kernel-compile function with
  in-memory + persistent ``.o`` caching (see :mod:`quack.cache.jit`).
* :data:`CACHE_ENABLED`, :data:`CACHE_DIR`, :data:`EXTRA_SOURCE_DIRS` —
  static-config flags.
* :class:`FileLock`, :func:`get_cache_path`, :class:`CacheInfo` —
  supporting types.

Async compilation (see :mod:`quack.cache.async_compile`):
* :class:`CompilePending` — raised by ``jit_cache`` on a cold miss while a
  compile pool is active; the caller defers and retries once the ``.o``
  lands.
* :func:`pool_scope` — activate a compile pool for a scoped block (used by
  the autotuner's bench loop).

CRITICAL ORDERING: the static-config flags below MUST be defined before the
``from quack.cache.jit import ...`` block. ``quack/cache/jit.py`` does
``import quack.cache as _state`` at its module top; Python returns the
partially-initialized package object, and lookups inside ``jit_cache``'s
wrapper rely on these names already existing at that checkpoint. Reordering
the imports here, even via an auto-formatter, will break the first kernel
compile with ``AttributeError``. The defensive unit tests in
``tests/test_cache.py`` exercise this path end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: Optional[str] = os.getenv("QUACK_CACHE_DIR", None)

#: Downstream projects can append directories here to include their sources
#: in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: List[Path] = []


# ---------------------------------------------------------------------------
# Public API surface. Imported AFTER the flags are defined.
# ---------------------------------------------------------------------------

from quack.cache.jit import (  # noqa: E402
    EXPORT_FUNC_NAME,
    LOCK_TIMEOUT,
    CacheInfo,
    FileLock,
    get_cache_path,
    jit_cache,
)
from quack.cache.async_compile import (  # noqa: E402
    CompilePending,
    pool_scope,
)

__all__ = [
    # Persistent .o cache.
    "jit_cache",
    "CacheInfo",
    "EXPORT_FUNC_NAME",
    "LOCK_TIMEOUT",
    "FileLock",
    "get_cache_path",
    # Async compilation.
    "CompilePending",
    "pool_scope",
]

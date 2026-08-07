# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inspecting and clearing the HostJIT precompiled-header cache.

The cache itself is populated lazily by the first build that needs an entry —
there is no warm-up. Generating an entry costs a few seconds and happens once
per (install, architecture, flag-set), so front-loading it was not worth the
machinery: doing it at import meant initializing the CUDA driver as a side
effect of importing this package, which is a far larger behavioral change than
the one-time cost it avoided.

Environment:
    CCCL_ENABLE_PCH=0             Disable precompiled headers entirely.
    CCCL_PCH_CACHE_DIR            Cache location; overrides the default chain.
    CCCL_PCH_CACHE_MAXSIZE        Size cap, LRU eviction (default 1 GiB, 0 off).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def cache_dir() -> Path | None:
    """Where the precompiled-header cache lives, or None if there is none.

    The location depends on ``CCCL_PCH_CACHE_DIR``, ``XDG_CACHE_HOME``, and the
    platform, so ask rather than assume. Returns None on the v1 (NVRTC) backend,
    which has no PCH cache, and when no writable location could be resolved.
    """
    try:
        from ._bindings import pch_cache_dir  # type: ignore[attr-defined]
    except ImportError:
        return None
    path = pch_cache_dir()
    return Path(path) if path else None


def clear_cache() -> int:
    """Delete every cached precompiled header. Returns the number of files removed.

    The cache is regenerated on demand, so this only costs the time to rebuild
    it (seconds per target). Use it to reclaim disk, or to force regeneration
    after changing something the cache key does not cover — notably an in-place
    CCCL header upgrade, which is otherwise detected only when the compiler
    rejects the stale entry and the build retries without it.

    Not synchronized against builds in other processes. That is safe on POSIX,
    where deleting a file another process has open leaves its handle valid; on
    Windows a file in use may simply refuse to delete, and is skipped. Either
    way the worst case is a rebuild, never a corrupt result.
    """
    directory = cache_dir()
    if directory is None or not directory.is_dir():
        return 0

    removed = 0
    for entry in directory.iterdir():
        try:
            if entry.is_dir():
                # Generation locks; a live one belongs to a build in flight, but
                # removing it only risks a duplicate generation, not corruption.
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
                removed += 1
        except OSError:
            # In use (Windows), or removed by someone else between listing and
            # unlinking. Neither is worth failing over.
            continue
    return removed

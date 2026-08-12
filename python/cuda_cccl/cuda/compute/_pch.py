# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Policy for the precompiled-header cache.

This module decides where the cache lives, whether it is enabled, and when it
is pruned. The backend generates and loads entries at the location it is given;
it chooses nothing, so a build writes only where this module points it.

The cache fills lazily: the first build needing an entry generates it, which
costs a few seconds and happens once per (install, architecture, flag-set).
Builds after that read it.

Environment:
    CCCL_ENABLE_PCH=0             Disable precompiled headers entirely.
    CCCL_PCH_CACHE_DIR            Cache location; overrides the default chain.
    CCCL_PCH_CACHE_MAXSIZE        Size cap, LRU eviction (default 1 GiB, 0 off).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_DEFAULT_MAX_BYTES = 1024**3  # 1 GiB
_SUFFIXES = {"k": 1024, "m": 1024**2, "g": 1024**3}

# How long a generation lock may sit before its holder is presumed dead. A lock
# is a directory created next to the entry it guards, released when generation
# finishes; one left by a killed process would otherwise suppress generation
# forever. The threshold only has to exceed a legitimate generation, which takes
# seconds.
_LOCK_STALE_SECONDS = 600

# How long a partially written entry may sit before it is presumed abandoned.
# Entries are written to a temp file and renamed into place, so a generation
# killed midway leaves one behind -- tens of megabytes each, one per attempt.
# The window is generous because a live generation must never be swept.
_TEMP_STALE_SECONDS = 3600

_FALSEY = {"0", "false", "off", "no"}


def _enabled() -> bool:
    """CCCL_ENABLE_PCH as a kill switch. Anything unset or unrecognized is on."""
    return os.environ.get("CCCL_ENABLE_PCH", "").strip().lower() not in _FALSEY


def _candidates() -> list[Path]:
    """Cache locations to try, best first.

    ``CCCL_PCH_CACHE_DIR`` is used verbatim with no subdirectory appended, so CI
    and tests get exactly the path they asked for. Otherwise this is a
    persistent cache of tens of megabytes, not scratch, so the system temp
    directory is a poor first choice: it is shared, and whoever creates it first
    owns it.
    """
    explicit = os.environ.get("CCCL_PCH_CACHE_DIR", "").strip()
    if explicit:
        return [Path(explicit)]

    out: list[Path] = []
    if sys.platform == "win32":
        if local := os.environ.get("LOCALAPPDATA", "").strip():
            out.append(Path(local) / "cccl" / "hostjit_pch")
        out.append(Path(tempfile.gettempdir()) / "hostjit_pch")
        return out

    if xdg := os.environ.get("XDG_CACHE_HOME", "").strip():
        out.append(Path(xdg) / "cccl" / "hostjit_pch")
    if home := os.environ.get("HOME", "").strip():
        out.append(Path(home) / ".cache" / "cccl" / "hostjit_pch")
    # uid-scoped, so two users on one machine cannot land on the same directory
    # and fight over its permissions.
    out.append(Path(tempfile.gettempdir()) / f"hostjit_pch_{os.getuid()}")
    return out


def _writable(directory: Path) -> bool:
    """Can we actually write here? Creating a directory does not prove it."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        if not directory.is_dir():
            return False
    probe = directory / ".cccl_write_probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


def resolve_cache_dir() -> Path | None:
    """The cache directory to build against, or None to build without a PCH.

    Resolved once per process. None means precompiled headers are off, either
    because they were disabled or because no candidate was writable. An unusable
    cache can never fail a build, only fail to speed one up.
    """
    if not _enabled():
        return None
    for directory in _candidates():
        try:
            if _writable(directory):
                return directory
        except OSError:
            continue
    return None


def _max_bytes() -> int:
    """CCCL_PCH_CACHE_MAXSIZE in bytes. 0 disables eviction.

    The default holds roughly a dozen configurations. CUDA's analogous
    CUDA_CACHE_MAXSIZE defaults to 256 MiB, but its entries are cubins measured
    in kilobytes; a single PCH here is tens of megabytes, so the same default
    would hold three configurations and thrash.
    """
    raw = os.environ.get("CCCL_PCH_CACHE_MAXSIZE", "").strip().lower()
    if not raw:
        return _DEFAULT_MAX_BYTES
    multiplier = 1
    if raw and raw[-1].isalpha():
        raw = raw.removesuffix("ib") if raw.endswith("ib") else raw
        if not raw or raw[-1] not in _SUFFIXES:
            return _DEFAULT_MAX_BYTES
        multiplier = _SUFFIXES[raw[-1]]
        raw = raw[:-1].strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_BYTES
    return max(0, value) * multiplier


def evict(exempt: set[Path] | None = None) -> int:
    """Trim the cache to its size cap, least-recently-used first.

    The bound is on total size rather than entry age: a dozen configurations all
    used within any age window still costs a gigabyte. Recency is mtime, which a
    build refreshes on the entries it uses.

    A precompiled header and its preamble are evicted together, since the header
    records the preamble as an input. Returns the number of bytes reclaimed.
    """
    directory = cache_dir()
    if directory is None or not directory.is_dir():
        return 0

    exempt = exempt or set()
    reclaimed = 0

    # Reclaim generation locks whose holder died. A live lock belongs to a build
    # in flight and must be left alone, so only clearly stale ones go.
    cutoff = time.time() - _LOCK_STALE_SECONDS
    for lock in directory.glob("*.lock"):
        try:
            if lock.is_dir() and lock.stat().st_mtime < cutoff:
                lock.rmdir()
        except OSError:
            continue

    # Temp files left by a generation that was killed before its rename.
    temp_cutoff = time.time() - _TEMP_STALE_SECONDS
    for tmp in directory.glob("*.tmp"):
        try:
            if tmp.is_file() and tmp.stat().st_mtime < temp_cutoff:
                size = tmp.stat().st_size
                tmp.unlink()
                reclaimed += size
        except OSError:
            continue

    # A preamble whose header is gone is dead weight -- it is only ever an input
    # to that header. Sweep those too, since nothing below would find them:
    # entries are enumerated by header, so an orphan is invisible to the scan
    # and would occupy the cache indefinitely.
    for preamble in directory.glob("*_preamble.cu"):
        pch = preamble.with_name(preamble.name[: -len("_preamble.cu")] + ".pch")
        if pch.exists():
            continue
        try:
            size = preamble.stat().st_size
            preamble.unlink()
            reclaimed += size
        except OSError:
            continue

    # Debris above is reclaimed unconditionally; a disabled size cap means "keep
    # every entry", not "let dead locks and orphans accumulate".
    cap = _max_bytes()
    if cap == 0:
        return reclaimed

    entries = []
    total = 0
    for pch in directory.glob("*.pch"):
        if pch in exempt:
            continue
        preamble = pch.with_name(pch.name[: -len(".pch")] + "_preamble.cu")
        try:
            size = pch.stat().st_size + (
                preamble.stat().st_size if preamble.exists() else 0
            )
            entries.append((pch.stat().st_mtime, size, pch, preamble))
        except OSError:
            continue
        total += size
    # Everything in the directory counts toward the cap, including entries this
    # build is using, so measure them even though they cannot be evicted.
    for pch in exempt:
        try:
            total += pch.stat().st_size
        except OSError:
            pass

    if total <= cap:
        return reclaimed

    for _mtime, _size, pch, preamble in sorted(entries):
        if total <= cap:
            break
        # Count only what actually went away. A file that refuses to delete is
        # still occupying the cache, so charging it against the total would
        # leave the cache over its cap while reporting otherwise.
        freed = 0
        for path in (pch, preamble):
            try:
                size = path.stat().st_size
                path.unlink()
                freed += size
            except OSError:
                continue
        total -= freed
        reclaimed += freed
    return reclaimed


def cache_dir() -> Path | None:
    """Where the precompiled-header cache lives, or None if there is none.

    Returns None on the v1 (NVRTC) backend, which has no PCH cache, and when no
    writable location could be resolved.
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

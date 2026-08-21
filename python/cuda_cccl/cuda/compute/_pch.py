# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Policy for the precompiled-header cache.

This module decides where the cache lives, whether it is enabled, and when it
is pruned.

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
import threading
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

# An entry touched this recently belongs to a build that is running or has just
# finished, and is never evicted. Without this, a cap smaller than one build's
# working set would delete the very entries that build just produced, and every
# build would regenerate them; the cap is exceeded instead of enforced. A build
# takes seconds, so the window only has to outlast one.
_ACTIVE_SECONDS = 120

_FALSEY = {"0", "false", "off", "no"}

# Guards the one-time resolution in ensure_configured against a race between
# concurrent first builds.
_configured = False
_config_lock = threading.Lock()


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
                # Clear debris up front, not only after a build. A lock left by a
                # killed process makes the entry it guards unbuildable until it is
                # reclaimed, so waiting until after the build would mean this
                # process builds without the very entry it was about to generate.
                _sweep_debris(directory)
                return directory
        except OSError:
            continue
    return None


def _backend_has_pch() -> bool:
    """True on the v2 (HostJIT) backend, the only one with a PCH cache."""
    try:
        from ._build_info import USING_V2  # type: ignore[import-not-found]
    except ImportError:
        return False
    return bool(USING_V2)


def _apply(directory: Path | None) -> None:
    """Record `directory` in the build config, or disable PCH when None. Never raises."""
    try:
        from ._bindings import set_pch_cache_dir  # type: ignore[attr-defined]
    except ImportError:
        return
    try:
        set_pch_cache_dir(str(directory) if directory else None)
    except Exception:
        pass


def ensure_configured() -> None:
    """Resolve the cache directory and record it in the build config, once per process.

    Runs on the first build, so importing the package has no filesystem side
    effects. Never raises: the cache is an optimization.
    """
    global _configured
    if _configured:
        return
    with _config_lock:
        if _configured:
            return
        if _backend_has_pch():
            _apply(resolve_cache_dir())
        _configured = True


def reconfigure(cache_dir: str | os.PathLike[str] | None = None) -> None:
    """Change the precompiled-header cache directory for this process.

    With no argument, re-resolves from the environment — the same chain the
    first build uses, so changing ``CCCL_PCH_CACHE_DIR`` / ``CCCL_ENABLE_PCH``
    and calling this takes effect immediately. With ``cache_dir`` given, that
    path is used verbatim (like ``CCCL_PCH_CACHE_DIR``), bypassing the chain.

    A test and power-user hook; ordinary use never needs it.
    """
    global _configured
    with _config_lock:
        if cache_dir is not None:
            _apply(Path(cache_dir))
        elif _backend_has_pch():
            _apply(resolve_cache_dir())
        else:
            _apply(None)
        _configured = True


def _sweep_debris(directory: Path) -> int:
    """Reclaim what a killed build leaves behind. Returns bytes freed.

    Anything still in use must survive, so age is the only safe discriminator:
    a live lock belongs to a build in flight, and a fresh temp file is an entry
    being written right now.
    """
    reclaimed = 0

    # A generation lock is a directory taken before writing an entry and removed
    # after. One whose holder died would suppress that entry forever.
    lock_cutoff = time.time() - _LOCK_STALE_SECONDS
    for lock in directory.glob("*.lock"):
        try:
            if lock.is_dir() and lock.stat().st_mtime < lock_cutoff:
                lock.rmdir()
        except OSError:
            continue

    # Entries are written to a temp file and renamed into place; a generation
    # killed midway leaves one behind, at tens of megabytes a time.
    temp_cutoff = time.time() - _TEMP_STALE_SECONDS
    for tmp in directory.glob("*.tmp"):
        try:
            if tmp.is_file() and tmp.stat().st_mtime < temp_cutoff:
                size = tmp.stat().st_size
                tmp.unlink()
                reclaimed += size
        except OSError:
            continue

    # A preamble whose header is gone is dead weight, but age-gate it like a temp
    # file: a live generation writes the preamble before its .pch, so a fresh
    # orphan may be one being generated right now.
    for preamble in directory.glob("*_preamble.cu"):
        pch = preamble.with_name(preamble.name[: -len("_preamble.cu")] + ".pch")
        if pch.exists():
            continue
        try:
            if preamble.stat().st_mtime >= temp_cutoff:
                continue
            size = preamble.stat().st_size
            preamble.unlink()
            reclaimed += size
        except OSError:
            continue

    return reclaimed


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


def evict() -> int:
    """Trim the cache to its size cap, least-recently-used first.

    The bound is on total size rather than entry age: a dozen configurations all
    used within any age window still costs a gigabyte. Recency is mtime, which a
    build refreshes on the entries it uses.

    An entry a build is using is never evicted, so a cap smaller than one
    build's working set is exceeded rather than enforced -- evicting it would
    delete what that build just produced and force the next one to regenerate.
    Such entries are identified by recency: a build refreshes the timestamp of
    every entry it uses, so anything touched within the last `_ACTIVE_SECONDS`
    belongs to a build that is running or has just finished.

    A precompiled header and its preamble are evicted together, since the header
    records the preamble as an input. Returns the number of bytes reclaimed.
    """
    directory = cache_dir()
    if directory is None or not directory.is_dir():
        return 0

    reclaimed = _sweep_debris(directory)

    # Debris is reclaimed unconditionally; a disabled size cap means "keep every
    # entry", not "let dead locks and orphans accumulate".
    cap = _max_bytes()
    if cap == 0:
        return reclaimed

    active_after = time.time() - _ACTIVE_SECONDS
    entries = []
    total = 0
    for pch in directory.glob("*.pch"):
        preamble = pch.with_name(pch.name[: -len(".pch")] + "_preamble.cu")
        try:
            stat = pch.stat()
            size = stat.st_size + (preamble.stat().st_size if preamble.exists() else 0)
        except OSError:
            continue
        # Protected entries still count toward the cap -- they occupy the disk
        # either way -- they are simply not candidates for removal.
        total += size
        if stat.st_mtime >= active_after:
            continue
        entries.append((stat.st_mtime, size, pch, preamble))

    if total <= cap:
        return reclaimed

    for _mtime, _size, pch, preamble in sorted(entries):
        if total <= cap:
            break
        # Delete the .pch first; only remove its preamble if that succeeded. A
        # surviving .pch whose preamble is gone is a poisoned entry -- clang
        # validates the preamble on load -- and on Windows the .pch may be held
        # open by another process and refuse to delete while the preamble does
        # not. Count only what actually went away, so a file that refuses to
        # delete still counts against the cap.
        try:
            freed = pch.stat().st_size
            pch.unlink()
        except OSError:
            continue
        try:
            psize = preamble.stat().st_size
            preamble.unlink()
            freed += psize
        except OSError:
            pass
        total -= freed
        reclaimed += freed
    return reclaimed


def cache_dir() -> Path | None:
    """Where the precompiled-header cache lives, or None if there is none.

    Returns None on the v1 (NVRTC) backend, which has no PCH cache, and when no
    writable location could be resolved.
    """
    ensure_configured()
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

    # Only HostJIT artifacts: CCCL_PCH_CACHE_DIR is taken verbatim, so it may
    # hold unrelated files that this must not touch.
    removed = 0
    for pch in directory.glob("*.pch"):
        preamble = pch.with_name(pch.name[: -len(".pch")] + "_preamble.cu")
        try:
            pch.unlink()
            removed += 1
        except OSError:
            # .pch in use (Windows); keep its preamble so the entry stays valid
            # rather than leaving a .pch that fails to validate on load.
            continue
        try:
            preamble.unlink()
            removed += 1
        except OSError:
            pass
    # Orphan preambles (their .pch is gone) and abandoned temps.
    for preamble in directory.glob("*_preamble.cu"):
        pch = preamble.with_name(preamble.name[: -len("_preamble.cu")] + ".pch")
        if pch.exists():
            continue  # its .pch survived (in use); leave the pair intact
        try:
            preamble.unlink()
            removed += 1
        except OSError:
            continue
    for tmp in directory.glob("*.tmp"):
        try:
            tmp.unlink()
            removed += 1
        except OSError:
            continue
    # Age-gate lock removal: a live-but-slow generation still holds its lock, and
    # deleting it would let a second writer collide on the same entry.
    lock_cutoff = time.time() - _LOCK_STALE_SECONDS
    for lock in directory.glob("*.lock"):  # generation locks are directories
        try:
            if lock.stat().st_mtime < lock_cutoff:
                shutil.rmtree(lock, ignore_errors=True)
        except OSError:
            continue
    return removed

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Precompiled-header cache behavior (v2 HostJIT backend only).

Most cases exercise the cache policy in ``cuda.compute._pch`` directly, pointing
it at a scratch directory with ``reconfigure`` and fabricating entries on disk;
only generation, reuse, corrupt-fallback, and concurrency drive a real build.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute as cc
from cuda.compute import _pch, clear_all_caches

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    not USING_V2, reason="precompiled headers are a v2 (HostJIT) feature"
)

_MIB = 1024 * 1024

# Permission-based cases cannot be expressed on every platform: Windows chmod
# does not make a directory unwritable, and root bypasses the check outright
# (CI runs as root, so this is the common case, not an exotic one).
IS_WINDOWS = sys.platform == "win32"
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.fixture(autouse=True)
def _pch_isolation(monkeypatch):
    """Clean, isolated PCH state per test; restore the lazy-init flag afterward."""
    monkeypatch.delenv("CCCL_PCH_CACHE_DIR", raising=False)
    monkeypatch.delenv("CCCL_ENABLE_PCH", raising=False)
    monkeypatch.delenv("CCCL_PCH_CACHE_MAXSIZE", raising=False)
    clear_all_caches()
    try:
        yield
    finally:
        clear_all_caches()
        # A later build must re-resolve from the (now-restored) environment
        # rather than keep writing into this test's deleted tmp_path.
        _pch._configured = False


@pytest.fixture
def cache(tmp_path):
    """A scratch PCH cache directory for this test."""
    _pch.reconfigure(str(tmp_path))
    return tmp_path


def pch_files(directory):
    return sorted(p.name for p in directory.glob("*.pch"))


def _make_entry(directory, name, *, size, age):
    """Fabricate a cache entry (sparse ``size``-byte .pch + preamble), backdated ``age`` seconds."""
    pch = directory / f"{name}.pch"
    with open(pch, "wb") as f:
        if size:
            f.seek(size - 1)
            f.write(b"\0")
    preamble = directory / f"{name}_preamble.cu"
    preamble.write_text("// preamble\n")
    when = time.time() - age
    for p in (pch, preamble):
        os.utime(p, (when, when))
    return pch, preamble


def fresh_build(dtype=np.int32):
    """One reduce build. ``clear_all_caches`` drops the in-process build-result
    cache so the call actually reaches the compiler (and the on-disk PCH)."""
    clear_all_caches()
    d_in = DeviceArray.from_numpy(np.arange(4, dtype=dtype))
    d_out = DeviceArray.empty(1, dtype)
    h_init = np.zeros(1, dtype=dtype)
    cc.make_reduce_into(d_in=d_in, d_out=d_out, op=cc.OpKind.PLUS, h_init=h_init)


# --- resolution policy (no build required) -----------------------------------


def test_resolve_uses_cache_dir_env(tmp_path, monkeypatch):
    """CCCL_PCH_CACHE_DIR is used verbatim, no subdirectory appended."""
    monkeypatch.setenv("CCCL_PCH_CACHE_DIR", str(tmp_path))
    assert _pch.resolve_cache_dir() == tmp_path


@pytest.mark.skipif(
    IS_WINDOWS,
    reason="LOCALAPPDATA is the Windows default; XDG_CACHE_HOME is not consulted there",
)
def test_resolve_falls_back_to_xdg(tmp_path, monkeypatch):
    """With CCCL_PCH_CACHE_DIR unset, XDG_CACHE_HOME takes over."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _pch.resolve_cache_dir() == tmp_path / "xdg" / "cccl" / "hostjit_pch"


def test_resolve_disabled_returns_none(tmp_path, monkeypatch):
    """CCCL_ENABLE_PCH=0 turns the cache off even with a valid directory."""
    monkeypatch.setenv("CCCL_PCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CCCL_ENABLE_PCH", "0")
    assert _pch.resolve_cache_dir() is None


def test_resolve_uncreatable_dir_returns_none(tmp_path, monkeypatch):
    """A cache path nested under a regular file (ENOTDIR) resolves to no cache."""
    blocker = tmp_path / "regular_file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("CCCL_PCH_CACHE_DIR", str(blocker / "cache"))
    assert _pch.resolve_cache_dir() is None


@pytest.mark.skipif(
    IS_WINDOWS or IS_ROOT,
    reason="chmod cannot make a directory unwritable on Windows or for root",
)
def test_unwritable_dir_disables_pch(tmp_path, monkeypatch):
    """A directory that exists but cannot be written resolves to no cache."""
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    monkeypatch.setenv("CCCL_PCH_CACHE_DIR", str(readonly))
    try:
        assert _pch.resolve_cache_dir() is None
    finally:
        readonly.chmod(0o700)


# --- sweeping and eviction (no build required) -------------------------------


def test_sweep_reclaims_abandoned_debris(cache):
    """A killed generation's lock and temp file are reclaimed once old enough."""
    lock = cache / "device_sm89_deadbeef.lock"
    lock.mkdir()
    temp = cache / "device_sm89_deadbeef.pch.tmp"
    temp.write_bytes(b"\0" * 1024)
    orphan = cache / "device_sm89_orphan_preamble.cu"  # preamble with no .pch
    orphan.write_text("// orphan\n")

    old = time.time() - 3 * 60 * 60  # past both windows (10 min lock, 1 hr temp)
    for entry in (lock, temp, orphan):
        os.utime(entry, (old, old))

    _pch.evict()

    assert not lock.exists(), "orphaned lock survived the sweep"
    assert not temp.exists(), "orphaned temp file survived the sweep"
    assert not orphan.exists(), "orphaned preamble survived the sweep"


def test_fresh_debris_is_left_alone(cache):
    """A lock or temp from a live generation is too young to sweep."""
    lock = cache / "device_sm89_live.lock"
    lock.mkdir()
    temp = cache / "device_sm89_live.pch.tmp"
    temp.write_bytes(b"\0" * 1024)

    _pch.evict()  # both are seconds old

    assert lock.exists(), "a live generation's lock was swept"
    assert temp.exists(), "a temp file being written was swept"


def test_size_cap_evicts_lru_entries(cache, monkeypatch):
    """A cap below the total evicts least-recently-used entries; recent (in-use) ones are exempt."""
    _make_entry(cache, "device_sm89_stale", size=60 * _MIB, age=24 * 3600)
    _make_entry(cache, "device_sm89_recent", size=60 * _MIB, age=0)
    monkeypatch.setenv("CCCL_PCH_CACHE_MAXSIZE", "100M")

    _pch.evict()

    assert not (cache / "device_sm89_stale.pch").exists(), "LRU entry survived the cap"
    assert not (cache / "device_sm89_stale_preamble.cu").exists(), (
        "evicted entry left its preamble behind"
    )
    assert (cache / "device_sm89_recent.pch").exists(), "a recent entry was evicted"


def test_cap_below_working_set_keeps_recent_entries(cache, monkeypatch):
    """A cap too small for the recent working set is exceeded, not enforced."""
    _make_entry(cache, "device_sm89_a", size=60 * _MIB, age=0)
    _make_entry(cache, "host_sm89_b", size=60 * _MIB, age=0)
    monkeypatch.setenv("CCCL_PCH_CACHE_MAXSIZE", "1M")

    _pch.evict()

    assert pch_files(cache) == ["device_sm89_a.pch", "host_sm89_b.pch"], (
        "a cap below the recent working set evicted its entries"
    )


def test_size_cap_zero_disables_eviction(cache, monkeypatch):
    """CCCL_PCH_CACHE_MAXSIZE=0 keeps everything, however stale."""
    _make_entry(cache, "device_sm89_stale", size=60 * _MIB, age=24 * 3600)
    monkeypatch.setenv("CCCL_PCH_CACHE_MAXSIZE", "0")

    _pch.evict()

    assert (cache / "device_sm89_stale.pch").exists(), "eviction ran despite maxsize=0"


# --- reporting and clearing (no build required) ------------------------------


def test_cache_dir_reports_location(cache):
    """pch_cache_dir() reports the configured cache directory."""
    assert _pch.cache_dir() == cache


def test_clear_cache_empties_and_reports(cache):
    """clear_pch_cache() removes entries and reports the count; empty clears to 0."""
    assert _pch.clear_cache() == 0, "cleared something from an empty cache"

    _make_entry(cache, "device_sm89_x", size=1024, age=0)
    _make_entry(cache, "host_sm89_y", size=1024, age=0)

    assert _pch.clear_cache() > 0, "clear reported removing nothing"
    assert pch_files(cache) == [], "entries survived clear_cache()"
    # Clearing is not destructive to the cache itself -- it still resolves here.
    assert _pch.cache_dir() == cache


# --- build integration (drives the compiler) ---------------------------------


def test_build_populates_and_reuses_cache(cache):
    """A cold build creates both PCHs; a later build reuses them by identity."""
    fresh_build()

    names = pch_files(cache)
    assert len(names) == 2, f"expected a device and a host PCH, got {names}"
    assert any(n.startswith("device_sm") for n in names), names
    assert any(n.startswith("host_sm") for n in names), names

    # Reuse is identity, not timestamp: a build refreshes the mtime of every
    # entry it uses, so regeneration (atomic rename onto a new inode) is what a
    # changed inode detects.
    before = {p.name: p.stat().st_ino for p in cache.glob("*.pch")}
    stale = time.time() - 48 * 3600
    for p in cache.glob("*.pch"):
        os.utime(p, (stale, stale))

    fresh_build()

    after = {p.name: p.stat().st_ino for p in cache.glob("*.pch")}
    assert before == after, "second build regenerated the cache instead of reusing it"
    for p in cache.glob("*.pch"):
        assert p.stat().st_mtime > stale, f"{p.name} was reused without recording it"


def test_corrupt_pch_falls_back(cache):
    """A PCH the frontend rejects must not fail the build."""
    fresh_build()
    entries = list(cache.glob("*.pch"))
    assert entries

    for p in entries:
        p.write_bytes(b"not a precompiled header")

    fresh_build()  # must retry without the rejected PCH rather than fail

    for p in entries:
        if p.exists():
            assert p.read_bytes() != b"not a precompiled header", (
                f"{p.name} was reused after being rejected"
            )


def test_disabled_generates_nothing(cache, monkeypatch):
    """CCCL_ENABLE_PCH=0 builds successfully and writes no cache."""
    monkeypatch.setenv("CCCL_ENABLE_PCH", "0")
    _pch.reconfigure()  # re-resolve from the environment -> disabled
    assert _pch.cache_dir() is None

    fresh_build()

    assert pch_files(cache) == [], "PCH was generated despite CCCL_ENABLE_PCH=0"


def test_uncreatable_cache_dir_still_builds(tmp_path, monkeypatch):
    """An unusable cache location degrades to building without a PCH."""
    blocker = tmp_path / "regular_file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("CCCL_PCH_CACHE_DIR", str(blocker / "cache"))
    _pch.reconfigure()  # re-resolve -> None (uncreatable)
    assert _pch.cache_dir() is None

    fresh_build()  # builds without a PCH, must succeed


def test_concurrent_cold_builds_generate_once(cache):
    """Concurrent cold builds generate one shared PCH, not one each.

    Each thread reduces a different dtype so its build is a distinct in-process
    key -- otherwise all but the first would hit the build-result cache and
    never reach the compiler.
    """
    clear_all_caches()
    dtypes = [np.int16, np.int32, np.int64, np.float32]
    errors: list[BaseException] = []
    # Release every thread into make_reduce_into together, so they genuinely
    # contend for the generation lock rather than finishing one after another.
    ready = threading.Barrier(len(dtypes))

    def worker(dtype):
        try:
            d_in = DeviceArray.from_numpy(np.arange(4, dtype=dtype))
            d_out = DeviceArray.empty(1, dtype)
            h_init = np.zeros(1, dtype=dtype)
            try:
                ready.wait(timeout=120)
            except threading.BrokenBarrierError:
                pass  # a sibling failed to arrive; build anyway
            cc.make_reduce_into(
                d_in=d_in, d_out=d_out, op=cc.OpKind.PLUS, h_init=h_init
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(dt,)) for dt in dtypes]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)

    assert not errors, errors
    assert not any(t.is_alive() for t in threads), "a build thread did not finish"

    names = pch_files(cache)
    assert len(names) == 2, f"expected one device and one host PCH, got {names}"
    assert list(cache.glob("*.lock")) == [], "a generation lock outlived its holder"


# Distinct dtypes so each build is a distinct in-process key and actually reaches
# the compiler rather than hitting the build-result cache.
_STRESS_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint32, np.float32]


def _run_reduce(dtype):
    d_in = DeviceArray.from_numpy(np.arange(4, dtype=dtype))
    d_out = DeviceArray.empty(1, dtype)
    h_init = np.zeros(1, dtype=dtype)
    cc.make_reduce_into(d_in=d_in, d_out=d_out, op=cc.OpKind.PLUS, h_init=h_init)


def _stress(cache, churn):
    """Run builds on many threads while `churn` runs on another. No error escapes."""
    clear_all_caches()
    errors: list[BaseException] = []
    stop = threading.Event()

    def churner():
        while not stop.is_set():
            try:
                churn()
            except BaseException as exc:  # noqa: BLE001 - surfaced to main thread
                errors.append(exc)
                return

    def build(dtype):
        try:
            _run_reduce(dtype)
        except BaseException as exc:  # noqa: BLE001 - surfaced to main thread
            errors.append(exc)

    churn_thread = threading.Thread(target=churner)
    churn_thread.start()
    builders = [threading.Thread(target=build, args=(dt,)) for dt in _STRESS_DTYPES]
    for t in builders:
        t.start()
    for t in builders:
        t.join(timeout=900)
    stop.set()
    churn_thread.join(timeout=60)

    assert not errors, errors
    assert not any(t.is_alive() for t in builders), "a build thread did not finish"


def test_builds_race_reconfigure(cache):
    """Builds must survive reconfigure() churn on another thread.

    reconfigure() rebinds the shared build config's cache-dir bytes while build
    threads read that pointer under nogil -- stresses that shared state for tears
    or a use-after-free.
    """
    _stress(cache, lambda: _pch.reconfigure(str(cache)))


def test_builds_race_clear_and_evict(cache):
    """clear_pch_cache()/evict() on another thread must not corrupt live builds.

    Deleting an entry while a build reads it must never leave a .pch without its
    preamble (a poisoned entry); the worst outcome is a rebuild.
    """

    def churn():
        _pch.clear_cache()
        _pch.evict()

    _stress(cache, churn)

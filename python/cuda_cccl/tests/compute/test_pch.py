# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Precompiled-header cache behavior (v2 HostJIT backend only).

Each test runs in a subprocess with CCCL_PCH_CACHE_DIR pointed at a tmp_path, so
nothing here touches the shared user cache and each case starts from a known
cache state. Subprocesses are also the only honest way to observe cold-cache
behavior, since this process has already imported cuda.compute and may have
populated the shared cache.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap
import time

import pytest

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    not USING_V2, reason="precompiled headers are a v2 (HostJIT) feature"
)


# tests/ root, so subprocesses can import _utils the way the suite does.
TESTS_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)

# Permission-based cases cannot be expressed on every platform: Windows chmod
# does not make a directory unwritable, and root bypasses the check outright
# (CI runs as root, so this is the common case, not an exotic one).
IS_WINDOWS = sys.platform == "win32"
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def subprocess_env(cache_dir, **overrides):
    """Environment for a child interpreter: scratch cache + importable _utils."""
    env = dict(os.environ)
    env["CCCL_PCH_CACHE_DIR"] = str(cache_dir)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{TESTS_ROOT}{os.pathsep}{existing}" if existing else TESTS_ROOT
    )
    for k, v in overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


def run_python(code: str, cache_dir, **env_overrides):
    """Run `code` in a fresh interpreter against a scratch PCH cache."""
    env = subprocess_env(cache_dir, **env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def pch_files(cache_dir):
    return sorted(p.name for p in cache_dir.glob("*.pch"))


# A minimal build, enough to drive the HostJIT compile path.
BUILD_SNIPPET = """
    import numpy as np
    from _utils.device_array import DeviceArray
    import cuda.compute as cc

    d_in = DeviceArray.from_numpy(np.arange(4, dtype=np.int32))
    d_out = DeviceArray.empty(1, np.int32)
    h_init = np.zeros(1, dtype=np.int32)
    cc.make_reduce_into(d_in=d_in, d_out=d_out, op=cc.OpKind.PLUS, h_init=h_init)
"""


def test_build_populates_cache(tmp_path):
    """A build with an empty cache creates both PCHs and leaves them alone next time."""
    proc = run_python(BUILD_SNIPPET, tmp_path)
    assert proc.returncode == 0, proc.stderr

    names = pch_files(tmp_path)
    assert len(names) == 2, f"expected a device and a host PCH, got {names}"
    assert any(n.startswith("device_sm") for n in names), names
    assert any(n.startswith("host_sm") for n in names), names

    # Reuse is identity, not timestamp. A build refreshes the mtime of every
    # entry it uses, so that cannot distinguish reuse from regeneration;
    # regeneration writes a new file through an atomic rename and so lands on a
    # different inode.
    before = {p.name: p.stat().st_ino for p in tmp_path.glob("*.pch")}
    stale = time.time() - 48 * 3600
    for p in tmp_path.glob("*.pch"):
        os.utime(p, (stale, stale))

    proc = run_python(BUILD_SNIPPET, tmp_path)
    assert proc.returncode == 0, proc.stderr

    after = {p.name: p.stat().st_ino for p in tmp_path.glob("*.pch")}
    assert before == after, "second build regenerated the cache instead of reusing it"
    # Recency drives eviction order, so a reused entry must not keep looking old.
    for p in tmp_path.glob("*.pch"):
        assert p.stat().st_mtime > stale, f"{p.name} was reused without recording it"


def test_corrupt_pch_falls_back(tmp_path):
    """A PCH the frontend rejects must not fail the build."""
    proc = run_python(BUILD_SNIPPET, tmp_path)
    assert proc.returncode == 0, proc.stderr
    entries = list(tmp_path.glob("*.pch"))
    assert entries

    for p in entries:
        p.write_bytes(b"not a precompiled header")

    proc = run_python(BUILD_SNIPPET, tmp_path)
    assert proc.returncode == 0, (
        "build failed against a corrupt PCH instead of retrying without it:\n"
        + proc.stderr
    )
    # The rejected entries are dropped so the next build regenerates them rather
    # than tripping over the same files forever.
    for p in entries:
        if p.exists():
            assert p.read_bytes() != b"not a precompiled header", (
                f"{p.name} was reused after being rejected"
            )


def test_uncreatable_cache_dir_still_builds(tmp_path):
    """An unusable cache location degrades to building without a PCH.

    The cache path is nested under a regular file, so creating it fails with
    ENOTDIR. Unlike a chmod, that holds for any uid and on Windows, so this
    covers the degradation path in CI (which runs as root) as well.
    """
    blocker = tmp_path / "regular_file"
    blocker.write_text("not a directory")

    proc = run_python(BUILD_SNIPPET, blocker / "cache")
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(
    IS_WINDOWS or IS_ROOT,
    reason="chmod cannot make a directory unwritable on Windows or for root",
)
def test_unwritable_cache_dir_still_builds(tmp_path):
    """A cache directory that exists but cannot be written to is also survivable.

    This is the permission-denied counterpart to the ENOTDIR case above: it
    reaches the write probe, where the directory exists but cannot be written.
    """
    cache = tmp_path / "readonly"
    cache.mkdir()
    cache.chmod(0o500)
    try:
        proc = run_python(BUILD_SNIPPET, cache)
        assert proc.returncode == 0, proc.stderr
    finally:
        cache.chmod(0o700)


def test_enable_pch_0_disables(tmp_path):
    """CCCL_ENABLE_PCH=0 disables the cache outright."""
    proc = run_python(BUILD_SNIPPET, tmp_path, CCCL_ENABLE_PCH="0")
    assert proc.returncode == 0, proc.stderr
    assert pch_files(tmp_path) == [], "PCH was generated despite CCCL_ENABLE_PCH=0"


def test_concurrent_cold_builds_generate_once(tmp_path):
    """Concurrent cold builds must not each generate their own copy.

    Generation costs seconds and tens of megabytes, and simultaneous cold
    starts are routine (any test runner fanning out workers). One process
    should win the lock and generate; the rest build without a PCH rather than
    duplicating the work or waiting on it.
    """
    env = subprocess_env(tmp_path)

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(BUILD_SNIPPET)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    # A failing assert mid-loop would otherwise strand the remaining children,
    # which keep running a full build well past the end of the test.
    try:
        for p in procs:
            _, err = p.communicate(timeout=900)
            assert p.returncode == 0, err
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()

    names = pch_files(tmp_path)
    assert len(names) == 2, f"expected exactly one device and one host PCH, got {names}"
    # Locks are released on exit, not left behind to block later generations.
    assert list(tmp_path.glob("*.lock")) == [], "a generation lock outlived its holder"


def test_abandoned_generation_debris_is_swept(tmp_path):
    """Debris from a killed generation must not outlive its window.

    A build interrupted partway through generation leaves its lock directory and
    a partially written temp file behind, because the release that would have
    cleaned them up never runs. Both are reclaimed by the pruning pass once they
    are old enough to be presumed abandoned. Orphaned temps are tens of
    megabytes each, one per abandoned attempt.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(BUILD_SNIPPET)],
        env=subprocess_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(2.0)  # far enough in to be generating, well short of finishing
    proc.kill()
    proc.communicate(timeout=60)

    locks = list(tmp_path.glob("*.lock"))
    temps = list(tmp_path.glob("*.tmp"))
    if not locks and not temps:
        pytest.skip("build did not reach the lock/temp stage before being killed")

    # Age it past the reclaim windows (10 min for locks, 1 hour for temps).
    old = time.time() - 3 * 60 * 60
    for entry in locks + temps:
        os.utime(entry, (old, old))

    # Any subsequent build prunes the cache once it completes.
    proc = run_python(BUILD_SNIPPET, tmp_path)
    assert proc.returncode == 0, proc.stderr

    assert list(tmp_path.glob("*.lock")) == [], "orphaned lock survived the sweep"
    assert list(tmp_path.glob("*.tmp")) == [], "orphaned temp file survived the sweep"
    assert len(pch_files(tmp_path)) == 2, pch_files(tmp_path)


def test_size_cap_evicts_lru_entries(tmp_path):
    """A build evicts older entries to stay under CCCL_PCH_CACHE_MAXSIZE.

    Stands in for a second configuration (different include paths or CTK) whose
    entries this build should reclaim. Entries the build is about to compile
    against are exempt, so the cap can be exceeded by the working set of a
    single build — the same approximation ccache documents.
    """
    # A plausible-looking entry from an earlier, unrelated configuration.
    stale_pch = tmp_path / "device_sm89_0123456789abcdef.pch"
    stale_pch.write_bytes(b"\0" * (60 * 1024 * 1024))
    stale_preamble = tmp_path / "device_sm89_0123456789abcdef_preamble.cu"
    stale_preamble.write_text("// stale\n")
    old = time.time() - 24 * 60 * 60
    for entry in (stale_pch, stale_preamble):
        os.utime(entry, (old, old))

    # 100 MiB holds this build's ~83 MiB but not that plus the 60 MiB squatter.
    proc = run_python(BUILD_SNIPPET, tmp_path, CCCL_PCH_CACHE_MAXSIZE="100M")
    assert proc.returncode == 0, proc.stderr

    assert not stale_pch.exists(), "least-recently-used entry survived the cap"
    assert not stale_preamble.exists(), "evicted entry left its preamble behind"
    # The build's own entries are exempt and must still be usable.
    assert len(pch_files(tmp_path)) == 2, pch_files(tmp_path)


def test_size_cap_zero_disables_eviction(tmp_path):
    """CCCL_PCH_CACHE_MAXSIZE=0 keeps everything."""
    stale = tmp_path / "device_sm89_0123456789abcdef.pch"
    stale.write_bytes(b"\0" * (60 * 1024 * 1024))
    old = time.time() - 24 * 60 * 60
    os.utime(stale, (old, old))

    proc = run_python(BUILD_SNIPPET, tmp_path, CCCL_PCH_CACHE_MAXSIZE="0")
    assert proc.returncode == 0, proc.stderr
    assert stale.exists(), "eviction ran despite CCCL_PCH_CACHE_MAXSIZE=0"


def test_cache_dir_and_clear(tmp_path):
    """pch_cache_dir() reports the live location; clear_pch_cache() empties it."""
    code = """
        import numpy as np
        from _utils.device_array import DeviceArray
        import cuda.compute as cc

        assert cc.pch_cache_dir() is not None, "no cache directory resolved"
        assert cc.clear_pch_cache() == 0, "cleared something from an empty cache"

        d_in = DeviceArray.from_numpy(np.arange(4, dtype=np.int32))
        d_out = DeviceArray.empty(1, np.int32)
        h_init = np.zeros(1, dtype=np.int32)
        cc.make_reduce_into(d_in=d_in, d_out=d_out, op=cc.OpKind.PLUS, h_init=h_init)

        cache = cc.pch_cache_dir()
        assert list(cache.glob("*.pch")), "build left no PCH behind"

        assert cc.clear_pch_cache() > 0, "clear reported removing nothing"
        assert list(cache.glob("*.pch")) == [], "PCHs survived clear_pch_cache()"

        # Clearing is not destructive to the cache itself -- a later build just
        # regenerates.
        assert cc.pch_cache_dir() == cache
        print("ok")
    """
    proc = run_python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


CACHE_DIR_SNIPPET = "import cuda.compute as cc; print(cc.pch_cache_dir())"


def test_cache_dir_follows_env(tmp_path):
    """The reported directory tracks the resolution chain, not a fixed path."""
    proc = run_python(CACHE_DIR_SNIPPET, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path)


@pytest.mark.skipif(
    IS_WINDOWS,
    reason="LOCALAPPDATA is the Windows default; XDG_CACHE_HOME is not consulted there",
)
def test_cache_dir_falls_back_to_xdg(tmp_path):
    """With CCCL_PCH_CACHE_DIR unset, XDG_CACHE_HOME takes over."""
    env = subprocess_env(
        tmp_path, CCCL_PCH_CACHE_DIR=None, XDG_CACHE_HOME=str(tmp_path / "xdg")
    )
    proc = subprocess.run(
        [sys.executable, "-c", CACHE_DIR_SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path / "xdg" / "cccl" / "hostjit_pch")


def test_import_without_cuda_is_quiet(tmp_path):
    """With no visible device, importing must not warn, raise, or cache anything."""
    code = """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            import cuda.compute as cc
        print("ok")
    """
    proc = run_python(code, tmp_path, CUDA_VISIBLE_DEVICES="")
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
    assert pch_files(tmp_path) == [], "import touched the PCH cache"

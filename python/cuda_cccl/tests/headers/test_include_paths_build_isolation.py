# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Regression tests for header discovery under pip build isolation.

Pip build isolation strips the active venv's ``site-packages`` from
``sys.path`` even though ``cuda-cccl`` is still installed there. A discovery
fallback that scans only ``sys.path`` therefore fails to find the shipped
headers. ``iter_site_roots()`` must also consult the interpreter site
directories (``site.getsitepackages()`` / ``site.getusersitepackages()``) so
discovery still succeeds.
"""

from pathlib import Path

from cuda.cccl.headers.include_paths import iter_site_roots


def test_iter_site_roots_includes_site_packages_when_absent_from_sys_path(
    monkeypatch,
):
    """The venv site-packages must be discoverable via site dirs even when it
    has been stripped from ``sys.path`` (the pip build-isolation scenario)."""
    isolated_sys_path = ["/tmp/pip-build-env/overlay/lib/python3.x/site-packages"]
    venv_site_packages = "/venv/lib/python3.x/site-packages"

    monkeypatch.setattr("cuda.cccl.headers.include_paths.sys.path", isolated_sys_path)
    monkeypatch.setattr(
        "cuda.cccl.headers.include_paths.site.getsitepackages",
        lambda: [venv_site_packages],
    )
    monkeypatch.setattr(
        "cuda.cccl.headers.include_paths.site.getusersitepackages",
        lambda: "/home/user/.local/lib/python3.x/site-packages",
    )

    roots = list(iter_site_roots())

    assert Path(venv_site_packages).resolve() in roots
    # The sys.path entries are still scanned too.
    assert Path(isolated_sys_path[0]).resolve() in roots


def test_iter_site_roots_is_deduplicated(monkeypatch):
    """Roots present in both ``sys.path`` and the site dirs are yielded once."""
    shared = "/venv/lib/python3.x/site-packages"

    monkeypatch.setattr("cuda.cccl.headers.include_paths.sys.path", [shared])
    monkeypatch.setattr(
        "cuda.cccl.headers.include_paths.site.getsitepackages", lambda: [shared]
    )
    monkeypatch.setattr(
        "cuda.cccl.headers.include_paths.site.getusersitepackages", lambda: shared
    )

    roots = list(iter_site_roots())

    assert roots.count(Path(shared).resolve()) == 1


def test_iter_site_roots_tolerates_none_user_site(monkeypatch):
    """``getusersitepackages()`` can return ``None`` when user site is disabled
    (``python -s`` / ``PYTHONNOUSERSITE``); it must not raise."""
    monkeypatch.setattr("cuda.cccl.headers.include_paths.sys.path", ["/some/path"])
    monkeypatch.setattr(
        "cuda.cccl.headers.include_paths.site.getsitepackages", lambda: [None]
    )
    monkeypatch.setattr(
        "cuda.cccl.headers.include_paths.site.getusersitepackages", lambda: None
    )

    roots = list(iter_site_roots())

    assert Path("/some/path").resolve() in roots
    assert None not in roots


def test_iter_site_roots_tolerates_missing_getsitepackages(monkeypatch):
    """Some virtualenv setups lack ``getsitepackages``; it must be probed
    defensively rather than raising."""

    def _raise():
        raise AttributeError("getsitepackages is unavailable")

    monkeypatch.setattr("cuda.cccl.headers.include_paths.sys.path", ["/some/path"])
    monkeypatch.setattr("cuda.cccl.headers.include_paths.site.getsitepackages", _raise)
    monkeypatch.setattr(
        "cuda.cccl.headers.include_paths.site.getusersitepackages",
        lambda: "/home/user/.local/lib/python3.x/site-packages",
    )

    roots = list(iter_site_roots())

    assert Path("/some/path").resolve() in roots

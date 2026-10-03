# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import cuda.coop._headers as headers
from cuda.coop._headers import CoopIncludePaths, resolve_include_paths

_PACKAGE_ROOT = Path(__file__).parents[1]


def _write_source_checkout(checkout: Path) -> None:
    for path in (
        checkout / "thrust",
        checkout / "cub" / "cub",
        checkout / "cudax" / "include",
        checkout / "libcudacxx" / "include",
    ):
        path.mkdir(parents=True)
    (checkout / "cub" / "cub" / "version.cuh").write_text(
        "// source probe\n",
        encoding="utf-8",
    )


def test_environment_inside_checkout_uses_installed_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "cccl"
    _write_source_checkout(checkout)
    installed_module = (
        checkout
        / ".venv"
        / "lib"
        / "python3.14"
        / "site-packages"
        / "cuda"
        / "coop"
        / "_headers"
        / "__init__.py"
    )
    installed_module.parent.mkdir(parents=True)
    installed_module.touch()
    installed_bundle = tmp_path / "installed-bundle"
    installed_bundle.mkdir()
    expected = CoopIncludePaths(
        cccl=(installed_bundle,),
        cuda=(),
        origin="installed test bundle",
    )
    monkeypatch.setattr(headers, "_installed_include_paths", lambda: expected)

    paths = resolve_include_paths(start=installed_module)

    assert paths == expected


def test_source_resolution_uses_one_coherent_header_set() -> None:
    paths = resolve_include_paths(
        start=Path(__file__),
        required_headers=(
            "cub/block/block_load.cuh",
            "thrust/detail/raw_pointer_cast.h",
            "cuda/experimental/coop/algorithm",
            "cuda/std/cstdint",
        ),
    )
    checkout = _PACKAGE_ROOT.parents[1]
    assert paths.cccl == (
        checkout / "thrust",
        checkout / "cub",
        checkout / "cudax" / "include",
        checkout / "libcudacxx" / "include",
    )


def test_incomplete_source_checkout_reports_missing_headers(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "cccl"
    probe = checkout / "cub" / "cub" / "version.cuh"
    probe.parent.mkdir(parents=True)
    probe.touch()
    source_module = checkout / "python" / "cuda_coop" / "cuda" / "coop"
    source_module.mkdir(parents=True)

    with pytest.raises(
        headers.HeaderResolutionError, match="missing source include roots"
    ):
        resolve_include_paths(start=source_module)


def test_import_cuda_coop_does_not_import_cuda_bindings() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(_PACKAGE_ROOT)!r})
import cuda.coop
unexpected = sorted(
    name for name in sys.modules
    if name == "cuda.bindings" or name.startswith("cuda.bindings.")
)
if unexpected:
    raise RuntimeError(f"cuda.coop eagerly imported {{unexpected}}")
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "platform,configured", [("linux", False), ("win32", True)]
)
def test_cuda_header_fallback_respects_platform(
    tmp_path, monkeypatch, platform, configured
):
    from types import SimpleNamespace

    import cuda.pathfinder

    monkeypatch.setattr(headers, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        cuda.pathfinder, "find_nvidia_header_directory", lambda _: None
    )
    for name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    include = tmp_path / "CUDA Toolkit" / "include"
    if configured:
        include.mkdir(parents=True)
        (include / "cuda_runtime.h").touch()
        monkeypatch.setenv("CUDA_PATH", str(include.parent))
    calls = []
    select = headers._select_cuda_include_path

    def record(paths):
        paths = tuple(paths)
        calls.append(paths)
        # Avoid relying on this host's fallback toolkit installation.
        return select(path for path in paths if path == include)

    monkeypatch.setattr(headers, "_select_cuda_include_path", record)
    assert headers._cuda_include_paths() == ((include,) if configured else ())
    fallback = (Path("/usr/local/cuda/include"),) if platform == "linux" else ()
    assert calls[-1] == ((include,) if configured else ()) + fallback


def test_required_header_diagnostic_never_falls_back_to_toolkit() -> None:
    with pytest.raises(
        headers.HeaderResolutionError, match="does not fall back"
    ):
        resolve_include_paths(
            start=Path(__file__),
            required_headers=("cub/block/not_a_primitive.cuh",),
        )


def test_cuda_headers_are_required_only_when_compiling() -> None:
    paths = CoopIncludePaths(
        cccl=(Path("cccl/include"),), cuda=(), origin="test"
    )
    with pytest.raises(headers.HeaderResolutionError, match="cuda_runtime.h"):
        paths.as_tuple()

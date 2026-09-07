# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shutil
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import cuda.coop._headers as headers

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from ...support.paths import PACKAGE_ROOT


def _make_source_checkout(root: Path, *required_headers: str) -> None:
    for relative in (
        "thrust/thrust",
        "cub/cub",
        "cudax/include",
        "libcudacxx/include",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "cub/cub/version.cuh").write_text("// CUB version\n", encoding="utf-8")
    for header in required_headers:
        path = root / "cub" / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// required\n", encoding="utf-8")


def _make_cuda_include(root: Path) -> Path:
    include = root / "include"
    include.mkdir(parents=True)
    (include / "cuda_runtime.h").write_text("// CUDA runtime\n", encoding="utf-8")
    return include


def test_cuda_header_resolution_selects_one_pathfinder_root(monkeypatch, tmp_path):
    discovered = _make_cuda_include(tmp_path / "pathfinder")
    configured = _make_cuda_include(tmp_path / "configured")
    monkeypatch.setitem(
        sys.modules,
        "cuda.pathfinder",
        SimpleNamespace(
            find_nvidia_header_directory=lambda library: (
                str(discovered) if library == "cudart" else None
            )
        ),
    )
    monkeypatch.setenv("CUDA_HOME", str(configured.parent))

    assert headers._cuda_include_paths() == (discovered.resolve(),)


def test_cuda_header_resolution_uses_one_configured_fallback(monkeypatch, tmp_path):
    cuda_home = _make_cuda_include(tmp_path / "cuda-home")
    cuda_path = _make_cuda_include(tmp_path / "cuda-path")
    monkeypatch.setitem(
        sys.modules,
        "cuda.pathfinder",
        SimpleNamespace(find_nvidia_header_directory=lambda library: None),
    )
    monkeypatch.setenv("CUDA_HOME", str(cuda_home.parent))
    monkeypatch.setenv("CUDA_PATH", str(cuda_path.parent))

    assert headers._cuda_include_paths() == (cuda_path.resolve(),)


def test_compiler_include_tuple_rejects_missing_cuda_headers(tmp_path):
    malformed = tmp_path / "cuda/include"
    malformed.mkdir(parents=True)
    paths = headers.CoopIncludePaths(
        cccl=(tmp_path,),
        cuda=headers._select_cuda_include_path((malformed,)),
        origin="test",
    )

    with pytest.raises(
        headers.HeaderResolutionError,
        match="Configure CUDA_PATH or CUDA_HOME",
    ):
        paths.as_tuple()


def test_malformed_pathfinder_root_uses_configured_fallback(monkeypatch, tmp_path):
    malformed = tmp_path / "pathfinder/include"
    malformed.mkdir(parents=True)
    configured = _make_cuda_include(tmp_path / "configured")
    monkeypatch.setitem(
        sys.modules,
        "cuda.pathfinder",
        SimpleNamespace(
            find_nvidia_header_directory=lambda library: (
                str(malformed) if library == "cudart" else None
            )
        ),
    )
    monkeypatch.setenv("CUDA_PATH", str(configured.parent))

    assert headers._cuda_include_paths() == (configured.resolve(),)


def test_source_checkout_cccl_headers_precede_cuda_headers(monkeypatch, tmp_path):
    root = tmp_path / "cccl"
    _make_source_checkout(root, "cub/block/block_reduce.cuh")
    start = root / "python/cuda_coop/cuda/coop/cutlass"
    start.mkdir(parents=True)
    cuda_include = _make_cuda_include(tmp_path / "cuda")
    monkeypatch.setattr(headers, "_cuda_include_paths", lambda: (cuda_include,))

    resolved = headers.resolve_include_paths(
        start=start,
        required_headers=("cub/block/block_reduce.cuh",),
    )

    assert resolved.origin == f"CCCL source checkout {root}"
    assert root / "cub" in resolved.cccl
    assert resolved.as_tuple() == (*resolved.cccl, cuda_include)


def test_installed_bundle_remains_available_without_cuda(monkeypatch, tmp_path):
    package = tmp_path / "site-packages/cuda/coop/_headers"
    bundled = package / "include"
    (bundled / "cub").mkdir(parents=True)
    (bundled / "cub/version.cuh").write_text("", encoding="utf-8")
    monkeypatch.setattr(headers, "files", lambda package_name: package)
    monkeypatch.setattr(headers, "_cuda_include_paths", lambda: ())

    resolved = headers.resolve_include_paths(start=tmp_path / "outside")

    assert resolved.cccl == (bundled.resolve(),)
    assert resolved.cuda == ()
    with pytest.raises(headers.HeaderResolutionError, match="Configure CUDA_PATH"):
        resolved.as_tuple()


def test_missing_source_header_does_not_fall_back_to_toolkit_cub(monkeypatch, tmp_path):
    root = tmp_path / "cccl"
    _make_source_checkout(root)
    start = root / "python/cuda_coop/cuda/coop/cutlass"
    start.mkdir(parents=True)

    toolkit_include = tmp_path / "cuda/include"
    (toolkit_include / "cub/block").mkdir(parents=True)
    (toolkit_include / "cub/version.cuh").write_text("", encoding="utf-8")
    (toolkit_include / "cub/block/block_row_reduce.cuh").write_text(
        "", encoding="utf-8"
    )
    (toolkit_include / "cuda_runtime.h").write_text("", encoding="utf-8")
    monkeypatch.setattr(headers, "_cuda_include_paths", lambda: (toolkit_include,))

    with pytest.raises(
        headers.HeaderResolutionError,
        match="does not fall back to CUDA toolkit CUB headers",
    ):
        headers.resolve_include_paths(
            start=start,
            required_headers=("cub/block/block_row_reduce.cuh",),
        )


def test_configured_cuda_toolkit_is_not_accepted_as_cccl_root(tmp_path):
    toolkit = tmp_path / "cuda"
    include = toolkit / "include"
    (include / "cub").mkdir(parents=True)
    (include / "cub/version.cuh").write_text("", encoding="utf-8")
    (include / "cuda_runtime.h").write_text("", encoding="utf-8")

    with pytest.raises(headers.HeaderResolutionError, match="CUDA toolkit"):
        headers.resolve_include_paths(
            start=tmp_path / "outside",
            configured_roots=(toolkit,),
        )


def test_installed_cuda_coop_bundle_is_used_outside_a_source_checkout(
    monkeypatch, tmp_path
):
    package = tmp_path / "site-packages/cuda/coop/_headers"
    bundled = package / "include"
    (bundled / "cub/block").mkdir(parents=True)
    (bundled / "cub/version.cuh").write_text("", encoding="utf-8")
    (bundled / "cub/block/block_reduce.cuh").write_text("", encoding="utf-8")
    cuda_include = _make_cuda_include(tmp_path / "cuda")
    monkeypatch.setattr(headers, "files", lambda package_name: package)
    monkeypatch.setattr(headers, "_cuda_include_paths", lambda: (cuda_include,))

    resolved = headers.resolve_include_paths(
        start=tmp_path / "outside",
        required_headers=("cub/block/block_reduce.cuh",),
    )

    assert resolved == headers.CoopIncludePaths(
        cccl=(bundled,),
        cuda=(cuda_include,),
        origin="cuda-coop wheel header bundle",
    )


def test_extracted_installed_bundle_lives_for_resolved_path(monkeypatch, tmp_path):
    extracted = tmp_path / "extracted site packages" / "cuda/coop/_headers/include"
    cuda_include = _make_cuda_include(tmp_path / "CUDA Toolkit")

    @contextmanager
    def temporary_resource(_resource):
        (extracted / "cub/block").mkdir(parents=True)
        (extracted / "cub/version.cuh").write_text("", encoding="utf-8")
        (extracted / "cub/block/block_reduce.cuh").write_text("", encoding="utf-8")
        (extracted / "cuda/experimental").mkdir(parents=True)
        (extracted / "cuda/experimental/coop.cuh").write_text("", encoding="utf-8")
        try:
            yield extracted
        finally:
            shutil.rmtree(extracted.parent.parent.parent)

    contexts = ExitStack()
    monkeypatch.setattr(headers, "_INSTALLED_HEADER_CONTEXTS", contexts)
    monkeypatch.setattr(headers, "files", lambda package_name: Path("resource"))
    monkeypatch.setattr(headers, "as_file", temporary_resource)
    monkeypatch.setattr(headers, "_cuda_include_paths", lambda: (cuda_include,))

    with contexts:
        resolved = headers.resolve_include_paths(
            start=tmp_path / "outside",
            required_headers=(
                "cub/block/block_reduce.cuh",
                "cuda/experimental/coop.cuh",
            ),
        )

        assert " " in str(resolved.cccl[0])
        assert (resolved.cccl[0] / "cub/block/block_reduce.cuh").is_file()
        assert (resolved.cccl[0] / "cuda/experimental/coop.cuh").is_file()
        assert resolved.as_tuple() == (extracted.resolve(), cuda_include.resolve())
        if sys.platform == "win32":
            assert "\\" in str(resolved.cccl[0])
    assert not extracted.exists()


def test_header_resolver_is_included_in_cuda_coop_wheels():
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as stream:
        package_map = tomllib.load(stream)["tool"]["scikit-build"]["wheel"]["packages"]

    assert package_map == {"cuda/coop": "cuda/coop"}
    assert (PACKAGE_ROOT / "cuda/coop/__init__.py").is_file()


def test_cuda_coop_metadata_owns_its_header_runtime_dependencies():
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    project = metadata["project"]
    providers = metadata["tool"]["dynamic-metadata"]

    assert project["dependencies"] == [
        "cuda-core",
        "cuda-pathfinder>=1.2.3",
        "numpy",
        "typing_extensions>=4.12.0",
    ]
    assert project["dynamic"] == ["version"]
    assert providers == [{"provider": "scikit_build_core.metadata.setuptools_scm"}]

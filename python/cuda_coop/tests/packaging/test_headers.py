# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

import cuda.coop._headers as headers
from cuda.coop._headers import (
    CoopIncludePaths,
    HeaderResolutionError,
    resolve_include_paths,
)

_PACKAGE_ROOT = Path(__file__).parents[2]


def test_installed_header_resource_is_resolved_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class Resource:
        def joinpath(self, name: str) -> Path:
            calls.append(f"joinpath:{name}")
            return tmp_path

    class Stack:
        def enter_context(self, context):
            calls.append("enter_context")
            return context.__enter__()

    def package_files(package: str) -> Resource:
        calls.append(f"files:{package}")
        return Resource()

    monkeypatch.setattr(headers, "files", package_files)
    monkeypatch.setattr(headers, "as_file", nullcontext)
    monkeypatch.setattr(headers, "_INSTALLED_HEADER_CONTEXTS", Stack())
    headers._installed_header_root.cache_clear()
    try:
        assert headers._installed_header_root() == tmp_path
        assert headers._installed_header_root() == tmp_path
    finally:
        headers._installed_header_root.cache_clear()

    assert calls == [
        "files:cuda.coop._headers",
        "joinpath:include",
        "enter_context",
    ]


def test_source_resolution_uses_only_required_cccl_header_trees() -> None:
    paths = resolve_include_paths(
        start=Path(__file__),
        required_headers=(
            "cub/block/block_reduce.cuh",
            "thrust/detail/raw_pointer_cast.h",
            "cuda/std/cstdint",
        ),
    )

    assert paths.origin.startswith("CCCL source checkout ")
    assert [path.name for path in paths.cccl] == ["thrust", "cub", "include"]
    assert all("cudax" not in path.parts for path in paths.cccl)


def test_environment_inside_checkout_does_not_capture_source_headers(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "cccl"
    for path in (
        checkout / "thrust",
        checkout / "cub" / "cub",
        checkout / "libcudacxx" / "include",
    ):
        path.mkdir(parents=True)
    (checkout / "cub" / "cub" / "version.cuh").write_text(
        "// source probe\n", encoding="utf-8"
    )
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

    assert headers._find_source_checkout(installed_module) is None


def test_source_package_path_resolves_its_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "cccl"
    for path in (
        checkout / "thrust",
        checkout / "cub" / "cub",
        checkout / "libcudacxx" / "include",
    ):
        path.mkdir(parents=True)
    (checkout / "cub" / "cub" / "version.cuh").write_text(
        "// source probe\n", encoding="utf-8"
    )
    source_module = (
        checkout / "python" / "cuda_coop" / "cuda" / "coop" / "_headers" / "__init__.py"
    )
    source_module.parent.mkdir(parents=True)
    source_module.touch()

    source = headers._find_source_checkout(source_module)

    assert source is not None
    root, include_paths = source
    assert root == checkout
    assert include_paths == (
        checkout / "thrust",
        checkout / "cub",
        checkout / "libcudacxx" / "include",
    )


def test_required_header_diagnostic_never_falls_back_to_toolkit() -> None:
    with pytest.raises(HeaderResolutionError, match="does not fall back"):
        resolve_include_paths(
            start=Path(__file__),
            required_headers=("cub/block/not_a_primitive.cuh",),
        )


def test_cuda_headers_are_required_only_when_compiling() -> None:
    paths = CoopIncludePaths(
        cccl=(Path("/private/cccl/include"),),
        cuda=(),
        origin="test",
    )
    with pytest.raises(HeaderResolutionError, match="cuda_runtime.h"):
        paths.as_tuple()


def test_package_metadata_excludes_cudax() -> None:
    cmake = (_PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "cudax" not in cmake.lower()
    assert "cudax" not in pyproject.lower()
    assert "CCCLInstallRules" not in cmake

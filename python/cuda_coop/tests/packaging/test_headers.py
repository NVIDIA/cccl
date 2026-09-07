# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from cuda.coop import _headers
from cuda.coop._headers import (
    CoopIncludePaths,
    HeaderResolutionError,
    resolve_include_paths,
)

_PACKAGE_ROOT = Path(__file__).parents[2]


def test_installed_header_root_enters_resource_context_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    include = package / "include"
    include.mkdir(parents=True)
    entered: list[Path] = []
    contexts = ExitStack()

    @contextmanager
    def counting_as_file(resource: Path) -> Iterator[Path]:
        entered.append(resource)
        yield resource

    _headers._installed_header_root.cache_clear()
    monkeypatch.setattr(_headers, "files", lambda package_name: package)
    monkeypatch.setattr(_headers, "as_file", counting_as_file)
    monkeypatch.setattr(_headers, "_INSTALLED_HEADER_CONTEXTS", contexts)
    try:
        assert _headers._installed_header_root() == include
        assert _headers._installed_header_root() == include
        assert entered == [include]
    finally:
        _headers._installed_header_root.cache_clear()
        contexts.close()


def test_source_resolution_uses_one_coherent_cccl_header_set() -> None:
    paths = resolve_include_paths(
        start=Path(__file__),
        required_headers=(
            "cub/block/block_load.cuh",
            "cub/block/block_store.cuh",
            "thrust/detail/raw_pointer_cast.h",
            "cuda/experimental/coop.cuh",
            "cuda/std/cstdint",
        ),
    )

    assert paths.origin.startswith("CCCL source checkout ")
    assert tuple(path.relative_to(_PACKAGE_ROOT.parents[1]) for path in paths.cccl) == (
        Path("thrust"),
        Path("cub"),
        Path("cudax/include"),
        Path("libcudacxx/include"),
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


def test_package_metadata_includes_cudax_header_bundle() -> None:
    cmake = (_PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "CCCLInstallRules" in cmake
    assert "CCCL_ENABLE_CUDAX ON" in cmake
    assert '"../../cudax/LICENSE.TXT"' in pyproject


def test_gitless_archive_does_not_inherit_enclosing_repository_revision(
    tmp_path: Path,
) -> None:
    required_tools = ("cmake", "git")
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing_tools:
        pytest.skip(
            f"required packaging tools are unavailable: {', '.join(missing_tools)}"
        )

    outer_repository = tmp_path / "outer"
    source_root = outer_repository / "cccl-archive"
    package_root = source_root / "python" / "cuda_coop"
    install_rules = source_root / "cmake" / "CCCLInstallRules.cmake"
    build_root = tmp_path / "build"

    package_root.mkdir(parents=True)
    install_rules.parent.mkdir(parents=True)
    shutil.copyfile(_PACKAGE_ROOT / "CMakeLists.txt", package_root / "CMakeLists.txt")
    install_rules.touch()

    git_env = os.environ.copy()
    for name in (
        "GIT_CEILING_DIRECTORIES",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
    ):
        git_env.pop(name, None)
    git_env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
    )

    subprocess.run(
        ["git", "-c", "init.templateDir=", "init", "--quiet", outer_repository],
        check=True,
        env=git_env,
    )
    subprocess.run(
        [
            "git",
            "-C",
            outer_repository,
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=cuda-coop packaging test",
            "-c",
            "user.email=cuda-coop-packaging-test@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "--message=outer repository",
        ],
        check=True,
        env=git_env,
    )

    prefix = subprocess.run(
        ["git", "-C", source_root, "rev-parse", "--show-prefix"],
        check=True,
        env=git_env,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert prefix == "cccl-archive/"

    subprocess.run(
        [
            "cmake",
            "-S",
            package_root,
            "-B",
            build_root,
        ],
        check=True,
        env=git_env,
    )

    provenance = json.loads(
        (build_root / "cccl-bundle-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance == {"cccl_source_commit": "unknown"}

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


def _require_packaging_tools() -> None:
    required_tools = ("cmake", "git")
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing_tools:
        pytest.skip(
            f"required packaging tools are unavailable: {', '.join(missing_tools)}"
        )


def _isolated_git_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "GIT_CEILING_DIRECTORIES",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
    ):
        env.pop(name, None)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
    )
    return env


def _prepare_minimal_cccl_source(source_root: Path) -> Path:
    package_root = source_root / "python" / "cuda_coop"
    install_rules = source_root / "cmake" / "CCCLInstallRules.cmake"
    package_root.mkdir(parents=True)
    install_rules.parent.mkdir(parents=True)
    shutil.copyfile(_PACKAGE_ROOT / "CMakeLists.txt", package_root / "CMakeLists.txt")
    install_rules.touch()
    return package_root


def _initialize_git_repository(source_root: Path, env: dict[str, str]) -> str:
    subprocess.run(
        [
            "git",
            "-C",
            source_root,
            "-c",
            "init.templateDir=",
            "init",
            "--quiet",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", source_root, "add", "--all"],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            "git",
            "-C",
            source_root,
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=cuda-coop packaging test",
            "-c",
            "user.email=cuda-coop-packaging-test@example.invalid",
            "commit",
            "--quiet",
            "--message=initial source",
        ],
        check=True,
        env=env,
    )
    return subprocess.run(
        ["git", "-C", source_root, "rev-parse", "HEAD"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_cuda_coop(
    package_root: Path,
    build_root: Path,
    env: dict[str, str],
    *definitions: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cmake",
            "-S",
            package_root,
            "-B",
            build_root,
            *definitions,
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )


def _bundle_provenance(build_root: Path) -> dict[str, str]:
    return json.loads(
        (build_root / "cccl-bundle-provenance.json").read_text(encoding="utf-8")
    )


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


@pytest.mark.parametrize(
    "missing_root",
    (Path("thrust"), Path("cudax/include"), Path("libcudacxx/include")),
)
def test_incomplete_cccl_source_roots_fail_closed(
    tmp_path: Path,
    missing_root: Path,
) -> None:
    source_root = tmp_path / "partial-cccl"
    include_roots = (
        Path("thrust"),
        Path("cub"),
        Path("cudax/include"),
        Path("libcudacxx/include"),
    )
    for include_root in include_roots:
        if include_root != missing_root:
            (source_root / include_root).mkdir(parents=True)
    cub_probe = source_root / "cub" / _headers._CUB_PROBE
    cub_probe.parent.mkdir(parents=True, exist_ok=True)
    cub_probe.touch()

    with pytest.raises(HeaderResolutionError, match=missing_root.as_posix()):
        resolve_include_paths(
            start=source_root,
            configured_roots=(source_root,),
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
    _require_packaging_tools()

    outer_repository = tmp_path / "outer"
    source_root = outer_repository / "cccl-archive"
    package_root = _prepare_minimal_cccl_source(source_root)
    build_root = tmp_path / "build"

    git_env = _isolated_git_env()

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

    result = _configure_cuda_coop(package_root, build_root, git_env)
    result.check_returncode()

    assert _bundle_provenance(build_root) == {"cccl_source_commit": "unknown"}


@pytest.mark.parametrize("change", ("modified", "untracked", "deleted", "ignored"))
def test_changed_header_bundle_fails_closed(tmp_path: Path, change: str) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    header = source_root / "cub" / "cub" / "test.cuh"
    header.parent.mkdir(parents=True)
    if change in {"modified", "deleted"}:
        header.write_text("// original\n", encoding="utf-8")
    elif change == "ignored":
        (source_root / ".gitignore").write_text("/cub/cub/test.cuh\n", encoding="utf-8")

    git_env = _isolated_git_env()
    _initialize_git_repository(source_root, git_env)
    if change == "modified":
        header.write_text("// modified\n", encoding="utf-8")
    elif change == "deleted":
        header.unlink()
    else:
        header.write_text("// local\n", encoding="utf-8")

    result = _configure_cuda_coop(
        package_root,
        tmp_path / "build",
        git_env,
    )

    assert result.returncode != 0
    diagnostic = result.stdout + result.stderr
    assert "CCCL header bundle inputs contain local changes" in diagnostic
    assert "CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE=ON" in diagnostic


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("lib/cmake/cub/cub-config.cmake"),
        Path("cmake/install/cub.cmake"),
    ),
)
def test_changed_bundle_cmake_input_fails_closed(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    bundle_input = source_root / relative_path
    bundle_input.parent.mkdir(parents=True, exist_ok=True)
    bundle_input.write_text("# original\n", encoding="utf-8")
    git_env = _isolated_git_env()
    _initialize_git_repository(source_root, git_env)
    bundle_input.write_text("# modified\n", encoding="utf-8")

    result = _configure_cuda_coop(package_root, tmp_path / "build", git_env)

    assert result.returncode != 0
    assert "CCCL header bundle inputs contain local changes" in (
        result.stdout + result.stderr
    )


def test_changed_cuda_coop_cmake_input_fails_closed(tmp_path: Path) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    git_env = _isolated_git_env()
    _initialize_git_repository(source_root, git_env)
    with (package_root / "CMakeLists.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n# Local packaging experiment.\n")

    result = _configure_cuda_coop(package_root, tmp_path / "build", git_env)

    assert result.returncode != 0
    assert "CCCL header bundle inputs contain local changes" in (
        result.stdout + result.stderr
    )


def test_allow_dirty_header_bundle_forces_unknown_provenance(tmp_path: Path) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    header = source_root / "cub" / "cub" / "test.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("// original\n", encoding="utf-8")
    git_env = _isolated_git_env()
    _initialize_git_repository(source_root, git_env)
    header.write_text("// modified\n", encoding="utf-8")

    claimed_revision = "1" * 40
    build_root = tmp_path / "build"
    result = _configure_cuda_coop(
        package_root,
        build_root,
        git_env,
        "-DCUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE=ON",
        f"-DCUDA_COOP_CCCL_SOURCE_REVISION={claimed_revision}",
    )
    result.check_returncode()

    assert _bundle_provenance(build_root) == {"cccl_source_commit": "unknown"}
    diagnostic = result.stdout + result.stderr
    assert "recording an unknown" in diagnostic
    assert "CCCL source revision" in diagnostic


def test_source_revision_override_cannot_bypass_dirty_header_gate(
    tmp_path: Path,
) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    header = source_root / "cub" / "cub" / "test.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("// original\n", encoding="utf-8")
    git_env = _isolated_git_env()
    _initialize_git_repository(source_root, git_env)
    header.write_text("// modified\n", encoding="utf-8")

    result = _configure_cuda_coop(
        package_root,
        tmp_path / "build",
        git_env,
        f"-DCUDA_COOP_CCCL_SOURCE_REVISION={'2' * 40}",
    )

    assert result.returncode != 0
    assert "CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE=ON" in (result.stdout + result.stderr)


def test_unrelated_dirty_file_preserves_head_revision(tmp_path: Path) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    git_env = _isolated_git_env()
    revision = _initialize_git_repository(source_root, git_env)
    unrelated = source_root / "docs" / "notes.md"
    unrelated.parent.mkdir()
    unrelated.write_text("local notes\n", encoding="utf-8")

    build_root = tmp_path / "build"
    result = _configure_cuda_coop(package_root, build_root, git_env)
    result.check_returncode()

    assert _bundle_provenance(build_root) == {"cccl_source_commit": revision}


def test_gitless_archive_accepts_explicit_source_revision(tmp_path: Path) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl-archive"
    package_root = _prepare_minimal_cccl_source(source_root)
    git_env = _isolated_git_env()
    revision = "3" * 40
    build_root = tmp_path / "build"
    result = _configure_cuda_coop(
        package_root,
        build_root,
        git_env,
        f"-DCUDA_COOP_CCCL_SOURCE_REVISION={revision}",
    )
    result.check_returncode()

    assert _bundle_provenance(build_root) == {"cccl_source_commit": revision}

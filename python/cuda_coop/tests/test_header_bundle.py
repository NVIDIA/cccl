# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).parents[1]


def _require_packaging_tools() -> None:
    required_tools = ("cmake", "git")
    missing_tools = [
        tool for tool in required_tools if shutil.which(tool) is None
    ]
    if missing_tools:
        pytest.skip(
            "required packaging tools are unavailable: "
            + ", ".join(missing_tools)
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
    shutil.copyfile(
        _PACKAGE_ROOT / "CMakeLists.txt", package_root / "CMakeLists.txt"
    )
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

    result = _configure_cuda_coop(package_root, build_root, git_env)
    result.check_returncode()

    assert _bundle_provenance(build_root) == {"cccl_source_commit": "unknown"}


@pytest.mark.parametrize(
    "change,relative_path",
    [
        ("modified", Path("cub/cub/test.cuh")),
        ("untracked", Path("cudax/include/cuda/experimental/coop/algorithm")),
        ("ignored", Path("cudax/include/cuda/experimental/coop/group")),
        ("deleted", Path("cmake/install/cub.cmake")),
        ("modified", Path("python/cuda_coop/CMakeLists.txt")),
    ],
)
def test_changed_header_bundle_fails_closed(
    tmp_path: Path, change: str, relative_path: Path
) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    header = source_root / relative_path
    header.parent.mkdir(parents=True, exist_ok=True)
    if change in {"modified", "deleted"} and not header.exists():
        header.write_text("// original\n", encoding="utf-8")
    elif change == "ignored":
        (source_root / ".gitignore").write_text(
            f"/{relative_path.as_posix()}\n", encoding="utf-8"
        )

    git_env = _isolated_git_env()
    _initialize_git_repository(source_root, git_env)
    if change == "modified":
        with header.open("a", encoding="utf-8") as stream:
            stream.write("\n# modified\n")
    elif change == "deleted":
        header.unlink()
    else:
        header.write_text("// local\n", encoding="utf-8")

    result = _configure_cuda_coop(
        package_root,
        tmp_path / "build",
        git_env,
        f"-DCUDA_COOP_CCCL_SOURCE_REVISION={'2' * 40}",
    )

    assert result.returncode != 0
    diagnostic = result.stdout + result.stderr
    assert "CCCL header bundle inputs contain local changes" in diagnostic
    assert "CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE=ON" in diagnostic


def test_allow_dirty_header_bundle_forces_unknown_provenance(
    tmp_path: Path,
) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl"
    package_root = _prepare_minimal_cccl_source(source_root)
    header = source_root / "cub" / "cub" / "test.cuh"
    header.parent.mkdir(parents=True, exist_ok=True)
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


def test_gitless_archive_accepts_explicit_source_revision(
    tmp_path: Path,
) -> None:
    _require_packaging_tools()

    source_root = tmp_path / "cccl-archive"
    package_root = _prepare_minimal_cccl_source(source_root)
    git_env = _isolated_git_env()
    revision = "v1.2.3+archive"
    build_root = tmp_path / "build"
    result = _configure_cuda_coop(
        package_root,
        build_root,
        git_env,
        f"-DCUDA_COOP_CCCL_SOURCE_REVISION={revision}",
    )
    result.check_returncode()

    assert _bundle_provenance(build_root) == {"cccl_source_commit": revision}

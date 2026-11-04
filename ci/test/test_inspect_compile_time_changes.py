#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with python3 -m pytest ci/test/test_inspect_compile_time_changes.py."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect_changes  # noqa: E402
from inspect_compile_time_changes import compile_time_changed  # noqa: E402


def benchmark_selected(paths):
    config = inspect_changes.load_config(inspect_changes.CONFIG_PATH)
    ignored = inspect_changes.compile_patterns(config.ignore_regexes)
    dirty_files = [
        path for path in paths if not inspect_changes.matches_any(ignored, path)
    ]
    ignored_files = [
        path for path in paths if inspect_changes.matches_any(ignored, path)
    ]
    return compile_time_changed(config, dirty_files, ignored_files)


@pytest.mark.parametrize(
    "path",
    [
        "python/cuda_coop_cutlass/cuda/coop/cutlass/_thread_data.py",
        "python/cuda_coop_cutlass/pyproject.toml",
        "python/cuda_coop_cutlass/ci/matrix.yaml",
        "ci/test_cuda_coop_cutlass.sh",
        "python/cuda_cccl/cuda/compute/_bindings.py",
        "python/cuda_stf/cuda/stf/_experimental.py",
        "docs/python/coop_cutlass/programming_guide.rst",
        "cub/test/catch2_test_block_scan.cu",
        "ci/compile_time/README.md",
        "cudax/include/cuda/experimental/device",
    ],
)
def test_unrelated_changes_do_not_allocate_benchmark_runners(path):
    assert not benchmark_selected([path])


@pytest.mark.parametrize(
    "path",
    [
        "cub/cub/block/block_scan.cuh",
        "thrust/thrust/sort.h",
        "libcudacxx/include/cuda/std/tuple",
        "cub/cmake/CubBuildCompilerTargets.cmake",
        "thrust/CMakeLists.txt",
        "libcudacxx/cmake/LibcudacxxPublicHeaderTesting.cmake",
        "cccl-version.json",
        "CMakePresets.json",
        "ci/matrix.yaml",
        "ci/project_files_and_dependencies.yaml",
        ".devcontainer/launch.sh",
        "ci/future_shared_infrastructure.py",
    ],
)
def test_shared_headers_and_build_inputs_select_the_benchmark(path):
    assert benchmark_selected([path])


@pytest.mark.parametrize(
    "path",
    [
        "ci/compile_time/summarize_events.py",
        "ci/build_compile_time_bench.sh",
        ".github/workflows/compile-time-bench.yml",
    ],
)
def test_benchmark_inputs_select_the_benchmark(path):
    assert benchmark_selected([path])


@pytest.mark.parametrize(
    ("earlier_path", "expected"),
    [
        ("python/cuda_coop_cutlass/pyproject.toml", "false"),
        ("ci/compile_time/parse_matrix.py", "true"),
    ],
)
def test_refs_use_the_pr_merge_base_and_write_github_outputs(
    tmp_path, monkeypatch, earlier_path, expected
):
    # Upstream-only changes must not add jobs to a PR; earlier commits in the PR
    # must still count even when its final commit changes an unrelated file.
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    def commit_file(path, contents):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
        git("add", path)
        git("-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        return git("rev-parse", "HEAD")

    git("init", "-q")
    git("remote", "add", "origin", str(tmp_path))
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "CI routing test")
    base = commit_file("README.md", "base\n")
    upstream = commit_file("cub/cub/block/block_scan.cuh", "upstream only\n")
    git("checkout", "--detach", base)
    commit_file(earlier_path, "earlier PR commit\n")
    head = commit_file(
        "python/cuda_coop_cutlass/cuda/coop/cutlass/_thread_data.py",
        "latest PR commit\n",
    )

    monkeypatch.setattr(inspect_changes, "REPO_ROOT", tmp_path)
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    from inspect_compile_time_changes import main

    assert main(["--refs", upstream, head]) == 0
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
        "compile_time_changed": expected,
    }

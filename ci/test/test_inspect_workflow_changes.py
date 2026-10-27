#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with python3 -m pytest ci/test/test_inspect_workflow_changes.py."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect_changes  # noqa: E402
from inspect_workflow_changes import select_workflows  # noqa: E402

ALL_JOBS = {"rapids", "matx", "pytorch", "compile_time"}


def selected_jobs(paths):
    config = inspect_changes.load_config(inspect_changes.CONFIG_PATH)
    ignored = inspect_changes.compile_patterns(config.ignore_regexes)
    dirty_files = [
        path for path in paths if not inspect_changes.matches_any(ignored, path)
    ]
    ignored_files = [
        path for path in paths if inspect_changes.matches_any(ignored, path)
    ]
    return {
        job
        for job, enabled in select_workflows(config, dirty_files, ignored_files).items()
        if enabled
    }


@pytest.mark.parametrize(
    "path",
    [
        "python/cuda_coop/cuda/coop/numba_mlir/_types.py",
        "python/cuda_coop/cuda/coop/_core/dispatch.py",
        "python/cuda_coop/pyproject.toml",
        "ci/test_cuda_coop_python.sh",
        "python/cuda_cccl/cuda/compute/_bindings.py",
        "python/cuda_stf/cuda/stf/_experimental.py",
        "docs/python/coop/programming_guide.rst",
        "cub/test/catch2_test_block_scan.cu",
        "ci/compile_time/README.md",
    ],
)
def test_unrelated_changes_do_not_allocate_integration_runners(path):
    assert selected_jobs([path]) == set()


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
        ".devcontainer/launch.sh",
        "ci/future_shared_infrastructure.py",
    ],
)
def test_shared_headers_and_build_inputs_retain_integration_coverage(path):
    assert selected_jobs([path]) == ALL_JOBS


@pytest.mark.parametrize(
    ("path", "job"),
    [
        ("ci/rapids/rapids-entrypoint.sh", "rapids"),
        (".github/workflows/build-rapids.yml", "rapids"),
        ("ci/matx/build_matx.sh", "matx"),
        (".github/workflows/build-matx.yml", "matx"),
        ("ci/pytorch/build_pytorch.sh", "pytorch"),
        (".github/workflows/build-pytorch.yml", "pytorch"),
        ("ci/compile_time/summarize_events.py", "compile_time"),
        ("ci/build_compile_time_bench.sh", "compile_time"),
        (".github/workflows/compile-time-bench.yml", "compile_time"),
    ],
)
def test_workflow_inputs_select_their_own_job(path, job):
    assert selected_jobs([path]) == {job}


def test_cudax_headers_do_not_select_the_unrelated_public_header_benchmark():
    assert selected_jobs(["cudax/include/cuda/experimental/device"]) == ALL_JOBS - {
        "compile_time"
    }


def test_mixed_changes_keep_all_affected_jobs():
    assert selected_jobs(
        [
            "python/cuda_coop/cuda/coop/_core/dispatch.py",
            "ci/matx/build_matx.sh",
            "ci/compile_time/parse_matrix.py",
        ]
    ) == {"matx", "compile_time"}


def test_refs_use_the_pr_merge_base_and_write_github_outputs(tmp_path, monkeypatch):
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
    commit_file("ci/matx/build_matx.sh", "earlier PR commit\n")
    head = commit_file("python/cuda_coop/pyproject.toml", "latest PR commit\n")

    monkeypatch.setattr(inspect_changes, "REPO_ROOT", tmp_path)
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    from inspect_workflow_changes import main

    assert main(["--refs", upstream, head]) == 0
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
        "rapids_changed": "false",
        "matx_changed": "true",
        "pytorch_changed": "false",
        "compile_time_changed": "false",
    }

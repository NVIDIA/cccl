# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise cooperative job selection through the real matrix generator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def generate_matrix(tmp_path, full, lite=()):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/actions/workflow-build/build-workflow.py"),
            str(ROOT / "ci/matrix.yaml"),
            "--workflows",
            "pull_request",
            "--full-build-projects",
            *full,
            "--lite-build-projects",
            *lite,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    workflow = json.loads((tmp_path / "workflow/workflow.json").read_text())
    jobs = []
    for group in workflow.values():
        jobs.extend(group.get("standalone", []))
        for pair in group.get("two_stage", []):
            jobs.extend(pair["producers"])
            jobs.extend(pair["consumers"])
    return jobs


def test_host_contracts_do_not_allocate_gpu_runners(tmp_path):
    jobs = generate_matrix(tmp_path, ["cuda_coop_python"])
    assert jobs
    assert all(job["gpu_count"] == 0 for job in jobs)
    assert all("numba-mlir" not in job["command"] for job in jobs)


@pytest.mark.parametrize("lite", [False, True])
def test_numba_consumers_share_wheel_producers(tmp_path, lite):
    projects = ["cuda_coop_python", "cuda_coop_numba"]
    jobs = generate_matrix(tmp_path, [] if lite else projects, projects if lite else [])
    commands = [job["command"] for job in jobs]
    builds = [command for command in commands if "build_cuda_coop_python" in command]
    assert len(builds) == (1 if lite else 4)
    assert len(builds) == len(set(builds))
    assert sum("-stage numba-mlir-compile" in command for command in commands) == 1
    runtime = [job for job in jobs if "-stage numba-mlir-runtime" in job["command"]]
    assert len(runtime) == 1
    assert runtime[0]["runner"] == "linux-amd64-gpu-l4-latest-1"
    assert all(job["gpu_count"] == 0 for job in jobs if job not in runtime)
    for command in commands:
        script = command.split('"')[1]
        assert (ROOT / script).is_file(), command
        assert "cuda_coop_python" in script, command

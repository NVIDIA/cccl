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


@pytest.mark.parametrize("lite", [False, True])
def test_cutlass_gpu_allocation_waits_for_compiler_qualification(tmp_path, lite):
    projects = ["cuda_coop_python", "cuda_coop_cutlass", "cuda_coop_coexistence"]
    jobs = generate_matrix(tmp_path, [] if lite else projects, projects if lite else [])
    runtime = [job for job in jobs if "-stage cutlass-runtime" in job["command"]]
    assert len(runtime) == 1
    assert runtime[0]["runner"] == "linux-amd64-gpu-l4-latest-1"
    assert all(job["gpu_count"] == 0 for job in jobs if job not in runtime)
    assert not any("numba-mlir" in job["command"] for job in jobs)

    workflow = json.loads((tmp_path / "workflow/workflow.json").read_text())
    pairs = [pair for group in workflow.values() for pair in group.get("two_stage", [])]
    pair = next(pair for pair in pairs if runtime[0] in pair["consumers"])
    assert len(pair["producers"]) == 1
    producer = pair["producers"][0]
    assert "-stage cutlass-compile" in producer["command"]
    assert producer["gpu_count"] == 0
    assert producer["cuda"] == runtime[0]["cuda"]
    assert all((ROOT / job["command"].split('"')[1]).is_file() for job in jobs)


@pytest.mark.parametrize(
    "path,expected_gpu_stages,coexistence",
    [
        ("ci/cuda_coop_cutlass_requirements.txt", {"cutlass-runtime"}, True),
        (
            "python/cuda_coop/cuda/coop/cutlass/_compiler/_bundle.py",
            {"cutlass-runtime"},
            False,
        ),
        (
            "python/cuda_coop/cuda/coop/cutlass/_compiler/_activation.py",
            {"cutlass-runtime"},
            True,
        ),
        (
            "python/cuda_coop/cuda/coop/numba_mlir/_compiler/_activation.py",
            {"numba-mlir-runtime"},
            True,
        ),
        (
            "python/cuda_coop/cuda/coop/_core/api/_dispatch.py",
            {"numba-mlir-runtime", "cutlass-runtime"},
            True,
        ),
    ],
)
def test_backend_changes_select_only_their_gpu_stages(
    tmp_path, path, expected_gpu_stages, coexistence
):
    result = subprocess.run(
        [sys.executable, str(ROOT / "ci/inspect_changes.py"), "--stdin"],
        input=f"{path}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    selection = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("FULL_BUILD=", "LITE_BUILD="))
    )
    jobs = generate_matrix(
        tmp_path, selection["FULL_BUILD"].split(), selection["LITE_BUILD"].split()
    )
    stages = {
        job["command"].split("-stage ")[1].split()[0]
        for job in jobs
        if job["gpu_count"]
    }
    assert stages == expected_gpu_stages
    assert any("-stage contracts" in job["command"] for job in jobs)
    assert any("-stage coexistence" in job["command"] for job in jobs) == coexistence

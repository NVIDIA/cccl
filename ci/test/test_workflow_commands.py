# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared Bash commands through the real matrix generator and dispatcher.

Run with: python3 -m pytest ci/test/test_workflow_commands.py
"""

import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github/actions/workflow-build"
BASH = ["bash", "--noprofile", "--norc"]


@pytest.fixture(scope="module")
def matrix():
    return yaml.safe_load((REPO_ROOT / "ci/matrix.yaml").read_text())


def generate_workflow(matrix, tmp_path, workflow="nightly"):
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix))
    result = subprocess.run(
        [
            sys.executable,
            str(ACTION_DIR / "build-workflow.py"),
            str(matrix_path),
            "--workflows",
            workflow,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((tmp_path / "workflow/workflow.json").read_text())


@pytest.fixture
def generate(matrix, tmp_path):
    def run(**overrides):
        row = {
            "jobs": ["test_cpu"],
            "project": "thrust",
            "ctk": "13.4",
            "cxx": "msvc14.44",
            "std": 20,
            "sm": "89",
        }
        row.update(overrides)
        row = {key: value for key, value in row.items() if value is not None}
        custom_matrix = copy.deepcopy(matrix)
        custom_matrix["workflows"] = {
            "nightly": [row],
            "exclude": custom_matrix["workflows"]["exclude"],
        }
        return generate_workflow(custom_matrix, tmp_path)

    return run


def all_jobs(workflow):
    for group in workflow.values():
        yield from group.get("standalone", [])
        for chain in group.get("two_stage", []):
            yield from chain["producers"]
            yield from chain["consumers"]


@pytest.mark.parametrize(
    ("compiler", "os_name", "config_label"),
    [("msvc14.44", "windows", "MSVC14.44"), ("gcc14", "linux", "GCC14")],
)
def test_thrust_commands_preserve_native_dispatch(
    generate, tmp_path, matrix, compiler, os_name, config_label
):
    workflow = generate(cxx=compiler)
    (group,) = workflow.values()
    (chain,) = group["two_stage"]
    (producer,) = chain["producers"]
    (consumer,) = chain["consumers"]
    assert shlex.split(producer["command"]) == BASH + [
        "./ci/build_thrust.sh",
        "-std",
        "20",
        "-arch",
        "89",
    ]
    assert shlex.split(consumer["command"]) == BASH + [
        "./ci/test_thrust.sh",
        "-cpu-only",
        "-std",
        "20",
        "-arch",
        "89",
    ]
    label = f"[CTK13.4 {config_label} C++20]"
    assert producer["name"] == f"{label} Build(amd64): sm{{89}}"
    assert consumer["name"] == f"{label} TestCPU(amd64): sm{{89}}"
    assert producer["os"] == consumer["os"] == os_name
    assert producer["runner"] == consumer["runner"]
    assert producer["gpu_count"] == consumer["gpu_count"] == 0
    if os_name == "windows":
        assert producer["runner"] == "windows-amd64-cpu16"
        assert (
            producer["image"]
            == consumer["image"]
            == (
                f"rapidsai/devcontainers:{matrix['devcontainer_version']}-cl14.44-cuda13.4"
            )
        )

    # Artifact names use the distinct producer/consumer IDs, independent of shell.
    expected_ids = json.loads((tmp_path / "workflow/job_ids.json").read_text())
    assert producer["id"] != consumer["id"]
    assert set(expected_ids) == {producer["id"], consumer["id"]}
    subprocess.run(
        [
            sys.executable,
            str(ACTION_DIR / "prepare-workflow-dispatch.py"),
            "workflow/workflow.json",
        ],
        cwd=tmp_path,
        check=True,
    )
    dispatch = json.loads((tmp_path / "workflow/dispatch.json").read_text())
    native = dispatch.pop(f"{os_name}_two_stage")
    (key,) = native["keys"]
    (dispatched_chain,) = native["jobs"][key]
    assert dispatched_chain["producers"][0]["id"] == producer["id"]
    assert dispatched_chain["consumers"][0]["id"] == consumer["id"]
    assert all(not bucket["keys"] for bucket in dispatch.values())


@pytest.mark.parametrize("workflow_name", ["pull_request", "nightly", "weekly"])
def test_real_windows_matrix_uses_existing_shared_entrypoints(
    matrix, tmp_path, workflow_name
):
    workflow = generate_workflow(matrix, tmp_path, workflow_name)
    windows_jobs = [job for job in all_jobs(workflow) if job["os"] == "windows"]
    assert windows_jobs
    for job in windows_jobs:
        args = shlex.split(job["command"])
        assert args[:3] == BASH, job
        assert args[3].startswith("./ci/") and args[3].endswith(".sh"), job
        assert "/windows/" not in args[3], job
        assert (REPO_ROOT / args[3]).is_file(), job
        assert "-cxx" not in args, job
        assert job["runner"].startswith("windows-"), job
        assert "-cl" in job["image"], job


def test_cub_aliases_keep_launch_options(generate):
    workflow = generate(project="cub", jobs=["build"])
    commands = [shlex.split(job["command"]) for job in all_jobs(workflow)]
    assert sorted(commands) == sorted(
        BASH + ["./ci/build_cub.sh", flag, "-std", "20", "-arch", "89"]
        for flag in ["-no-lid", "-lid0", "-lid1", "-lid2"]
    )


def test_python_keeps_wheel_dependency_and_python_arguments(generate):
    workflow = generate(
        project="python",
        jobs=["test_headers"],
        std=None,
        sm=None,
        py_version="3.13",
        py_ctk_mode="latest",
    )
    (group,) = workflow.values()
    (chain,) = group["two_stage"]
    (producer,) = chain["producers"]
    (consumer,) = chain["consumers"]
    for job, entrypoint in [
        (producer, "build_cuda_cccl_python.sh"),
        (consumer, "test_cuda_cccl_headers_python.sh"),
    ]:
        expected = BASH + [f"./ci/{entrypoint}", "-py-version", "3.13"]
        if job is consumer:
            expected += ["-ctk-mode", "latest"]
        assert shlex.split(job["command"]) == expected
        assert job["os"] == "windows"
    assert "-cuda12.9" in producer["image"]
    assert "-cuda13.4" in consumer["image"]


@pytest.mark.parametrize("compiler", ["msvc14.44", "gcc14"])
def test_list_and_space_arguments_stay_single_arguments(generate, compiler):
    workflow = generate(
        cxx=compiler,
        sm="75;89",
        cmake_options="-DFIRST=ON -DSECOND=OFF",
        args='--configure-only --build-targets "first second"',
    )
    for job in all_jobs(workflow):
        args = shlex.split(job["command"])
        assert args[args.index("-arch") + 1] == "75;89"
        assert args[args.index("-cmake-options") + 1] == "-DFIRST=ON -DSECOND=OFF"
        assert args[-3:] == ["--configure-only", "--build-targets", "first second"]

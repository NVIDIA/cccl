# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shell selection through the real matrix generator and dispatcher.

Run with: python3 -m pytest ci/test/test_workflow_script_shell.py
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github/actions/workflow-build"


@pytest.fixture(scope="module")
def matrix():
    return yaml.safe_load((REPO_ROOT / "ci/matrix.yaml").read_text())


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
        experiment = copy.deepcopy(matrix)
        experiment["workflows"] = {
            "nightly": [row],
            "exclude": experiment["workflows"]["exclude"],
        }
        matrix_path = tmp_path / "matrix.yaml"
        matrix_path.write_text(yaml.safe_dump(experiment))
        return subprocess.run(
            [
                sys.executable,
                str(ACTION_DIR / "build-workflow.py"),
                str(matrix_path),
                "--workflows",
                "nightly",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    return run


def read_chains(tmp_path):
    workflow = json.loads((tmp_path / "workflow/workflow.json").read_text())
    return [chain for group in workflow.values() for chain in group["two_stage"]]


@pytest.mark.parametrize(
    ("compiler", "os_name", "script_dir", "script_ext", "config_label"),
    [
        ("msvc14.44", "windows", "./ci/windows", "ps1", "MSVC14.44"),
        ("gcc14", "linux", "./ci", "sh", "GCC14"),
    ],
)
def test_default_commands_and_names_are_unchanged(
    generate, tmp_path, compiler, os_name, script_dir, script_ext, config_label
):
    result = generate(cxx=compiler)
    assert result.returncode == 0, result.stderr
    (chain,) = read_chains(tmp_path)
    (producer,) = chain["producers"]
    (consumer,) = chain["consumers"]
    assert producer["command"] == (
        f'"{script_dir}/build_thrust.{script_ext}" -std "20" -arch "89"'
    )
    assert consumer["command"] == (
        f'"{script_dir}/test_thrust.{script_ext}" -cpu-only -std "20" -arch "89"'
    )
    label = f"[CTK13.4 {config_label} C++20]"
    assert producer["name"] == f"{label} Build(amd64): sm{{89}}"
    assert consumer["name"] == f"{label} TestCPU(amd64): sm{{89}}"
    assert producer["os"] == consumer["os"] == os_name


@pytest.mark.parametrize("job", ["test_cpu", "test"])
def test_shell_comparison_keeps_independent_native_windows_chains(
    generate, tmp_path, matrix, job
):
    result = generate(jobs=[job], script_shell=["powershell", "bash"])
    assert result.returncode == 0, result.stderr
    chains = read_chains(tmp_path)
    assert len(chains) == 2
    jobs = []
    for chain in chains:
        (producer,) = chain["producers"]
        (consumer,) = chain["consumers"]
        jobs.extend([producer, consumer])
        bash = producer["name"].endswith("[Bash]")
        shell_label = "Bash" if bash else "PowerShell"
        assert consumer["name"].endswith(f"[{shell_label}]")
        assert "TestCPU" in consumer["name"]
        assert producer["os"] == consumer["os"] == "windows"
        assert producer["runner"] == consumer["runner"] == "windows-amd64-cpu16"
        assert producer["gpu_count"] == consumer["gpu_count"] == 0
        image = (
            f"rapidsai/devcontainers:{matrix['devcontainer_version']}-cl14.44-cuda13.4"
        )
        assert producer["image"] == consumer["image"] == image
        if bash:
            prefix = "bash --noprofile --norc "
            assert producer["command"] == (
                f'{prefix}"./ci/build_thrust.sh" -cxx "cl" -std "20" -arch "89"'
            )
            assert consumer["command"] == (
                f'{prefix}"./ci/test_thrust.sh" -cxx "cl" -cpu-only -std "20" -arch "89"'
            )
        else:
            assert producer["command"] == (
                '"./ci/windows/build_thrust.ps1" -std "20" -arch "89"'
            )
            assert consumer["command"] == (
                '"./ci/windows/test_thrust.ps1" -cpu-only -std "20" -arch "89"'
            )

    # The artifact protocol uses these IDs to distinguish the two producers.
    ids = {job["id"] for job in jobs}
    assert len(ids) == 4
    expected_ids = json.loads((tmp_path / "workflow/job_ids.json").read_text())
    assert set(expected_ids) == ids
    assert len({job["name"] for job in jobs}) == 4

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
    windows = dispatch.pop("windows_two_stage")
    assert windows["keys"] == ["Thrust nvcc MSVC"]
    assert len(windows["jobs"]["Thrust nvcc MSVC"]) == 2
    assert all(not bucket["keys"] for bucket in dispatch.values())


@pytest.mark.parametrize("shell", ["sh", "", None, [], ["bash", "invalid"]])
def test_invalid_shell_choices_are_rejected(generate, shell):
    result = generate(script_shell=shell)
    assert result.returncode != 0
    assert "script_shell must be 'bash' or 'powershell'" in result.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        {"cxx": "gcc14"},
        {"project": "libcudacxx", "jobs": ["build"]},
        {"jobs": ["install"]},
    ],
)
def test_unported_script_selection_is_rejected(generate, overrides):
    result = generate(script_shell="bash", **overrides)
    assert result.returncode != 0
    assert (
        "script_shell is only supported for Windows Thrust build/test jobs."
        in result.stderr
    )

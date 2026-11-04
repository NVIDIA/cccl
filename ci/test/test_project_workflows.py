# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise project-owned workflow rows through the CI matrix generator."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / ".github/actions/workflow-build/build-workflow.py"
load_project_workflows = runpy.run_path(str(GENERATOR))["load_project_workflows"]


@pytest.fixture
def matrix(tmp_path):
    config = yaml.safe_load((ROOT / "ci/matrix.yaml").read_text())
    config["projects"]["fragment_test"] = {
        "name": "Fragment test",
        "workflows_file": "../package/workflows.yaml",
    }
    config["workflows"]["pull_request"] = [
        {"project": "cub", "jobs": ["build"], "ctk": "13.X", "cxx": "gcc13", "std": 17}
    ]
    matrix_path = tmp_path / "ci/matrix.yaml"
    matrix_path.parent.mkdir()
    return config, matrix_path


def generate(config, matrix_path):
    matrix_path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            str(matrix_path),
            "--workflows",
            "pull_request",
        ],
        cwd=matrix_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((matrix_path.parent / "workflow/workflow.json").read_text())


def write_fragment(matrix_path, workflows):
    path = matrix_path.parent.parent / "package/workflows.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(workflows, sort_keys=False))
    return path


def test_missing_project_workflows_preserve_existing_jobs(matrix):
    config, matrix_path = matrix
    actual = generate(config, matrix_path)
    del config["projects"]["fragment_test"]["workflows_file"]
    assert actual == generate(config, matrix_path)


def test_project_workflows_append_jobs_with_their_source_location(matrix):
    config, matrix_path = matrix
    path = write_fragment(
        matrix_path,
        {
            "pull_request": [
                {
                    "project": "fragment_test",
                    "jobs": ["build"],
                    "ctk": "13.X",
                    "cxx": "gcc13",
                    "std": 17,
                }
            ]
        },
    )
    workflow = generate(config, matrix_path)
    jobs = [job for group in workflow.values() for job in group.get("standalone", [])]
    assert {job["origin"]["matrix_job"]["project"] for job in jobs} == {
        "cub",
        "fragment_test",
    }
    fragment_job = next(
        job for job in jobs if job["origin"]["matrix_job"]["project"] == "fragment_test"
    )
    assert Path(fragment_job["origin"]["filename"]).resolve() == path.resolve()


@pytest.mark.parametrize(
    "workflows",
    [
        {"override": []},
        {"exclude": []},
        {"python-wheels": []},
        {"unknown": []},
        {"pull_request": [{"project": "cub"}]},
        {"pull_request": [{"project": ["fragment_test", "cub"]}]},
    ],
)
def test_project_workflows_cannot_change_other_projects_or_dispatch(matrix, workflows):
    config, matrix_path = matrix
    write_fragment(matrix_path, workflows)
    with pytest.raises(
        ValueError, match="unsupported project workflow|must select only project"
    ):
        load_project_workflows(config, matrix_path)

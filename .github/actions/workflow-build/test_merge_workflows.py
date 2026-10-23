# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import runpy
from pathlib import Path

merge_workflows = runpy.run_path(str(Path(__file__).with_name("merge-workflows.py")))[
    "merge_workflows"
]


def job(name, *, gpu=False, environment="[]"):
    return {
        "id": name,
        "name": name,
        "runner": "gpu-runner" if gpu else "cpu-runner",
        "image": "image",
        "environment": environment,
        "command": name,
        "origin": {"filename": "matrix.yaml"},
        "gpu_count": int(gpu),
    }


def write_workflow(path, workflow, overrides=None):
    path.mkdir()
    (path / "workflow.json").write_text(json.dumps(workflow))
    if overrides is not None:
        (path / "override.json").write_text(json.dumps(overrides))
    return path


def test_shared_producers_and_standalone_promotion(tmp_path):
    build, wheel, lint = job("build"), job("wheel"), job("lint")
    first_test = job("first-test", gpu=True)
    second_test = job("second-test", gpu=True)
    third_test = job("third-test", gpu=True)
    public = write_workflow(
        tmp_path / "public",
        {
            "group": {
                "standalone": [build, lint],
                "two_stage": [
                    {"id": "local", "producers": [wheel], "consumers": [first_test]}
                ],
            }
        },
    )
    auxiliary = write_workflow(
        tmp_path / "auxiliary",
        {
            "group": {
                "two_stage": [
                    {"id": "local", "producers": [build], "consumers": [second_test]},
                    {
                        "id": "other",
                        "producers": [wheel],
                        "consumers": [first_test, third_test],
                    },
                ]
            }
        },
    )
    output = tmp_path / "workflow"
    merge_workflows([auxiliary, public], output)
    result = json.loads((output / "workflow.json").read_text())["group"]

    assert [entry["name"] for entry in result["standalone"]] == ["lint"]
    consumers = {
        pair["producers"][0]["name"]: {entry["name"] for entry in pair["consumers"]}
        for pair in result["two_stage"]
    }
    assert consumers == {
        "build": {"second-test"},
        "wheel": {"first-test", "third-test"},
    }
    jobs = result["standalone"] + [
        entry
        for pair in result["two_stage"]
        for entry in pair["producers"] + pair["consumers"]
    ]
    ids = [entry["id"] for entry in jobs] + [pair["id"] for pair in result["two_stage"]]
    assert len(ids) == len(set(ids))
    assert set(json.loads((output / "job_ids.json").read_text())) == {
        entry["id"] for entry in jobs
    }
    summary = json.loads((output / "runner_summary.json").read_text())
    assert "total jobs: 6" in summary["heading"]
    assert "cpu-runner" in summary["body"] and "gpu-runner" in summary["body"]
    assert "first-test" in (output / "job_list.txt").read_text()


def test_same_name_different_execution_is_preserved(tmp_path):
    original = job("build")
    changed = copy.deepcopy(original)
    changed["environment"] = '["CI_MATRIX=auxiliary"]'
    paths = [
        write_workflow(tmp_path / name, {"group": {"standalone": [entry]}})
        for name, entry in [("public", original), ("auxiliary", changed)]
    ]
    output = tmp_path / "workflow"
    merge_workflows(paths, output)
    entries = json.loads((output / "workflow.json").read_text())["group"]["standalone"]
    assert {entry["environment"] for entry in entries} == {
        "[]",
        '["CI_MATRIX=auxiliary"]',
    }
    assert len({entry["id"] for entry in entries}) == 2


def test_overrides_are_collected_without_overwriting_change_detection(tmp_path):
    public = write_workflow(tmp_path / "public", {}, [{"project": "cub"}])
    auxiliary = write_workflow(tmp_path / "auxiliary", {}, [{"project": "thrust"}])
    output = tmp_path / "workflow"
    output.mkdir()
    (output / "changes.md").write_text("Change detection report")
    merge_workflows([public, auxiliary], output)
    assert json.loads((output / "override.json").read_text()) == [
        {"project": "cub"},
        {"project": "thrust"},
    ]
    assert (output / "changes.md").read_text() == "Change detection report"

    (public / "override.json").unlink()
    (auxiliary / "override.json").unlink()
    merge_workflows([public, auxiliary], output)
    assert not (output / "override.json").exists()
    assert json.loads((output / "workflow.json").read_text()) == {}
    assert json.loads((output / "job_ids.json").read_text()) == {}

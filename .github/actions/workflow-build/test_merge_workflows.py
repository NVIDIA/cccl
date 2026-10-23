# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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


@pytest.mark.skipif(
    sys.platform != "linux", reason="The composite action runs in Bash on Linux."
)
@pytest.mark.parametrize(
    "matrix_files",
    [
        ["ci/matrix.yaml", "ci/extra.yaml"],
        ["ci/matrix.yaml", "./ci/extra.yaml"],
        ["ci/2/matrix.yaml", "ci/matrix.yaml", "ci/extra.yaml"],
        ["workflow/standard/matrix.yaml", "workflow/extended/matrix.yaml"],
    ],
)
def test_action_output_directories(tmp_path, matrix_files):
    for index, matrix_file in enumerate(matrix_files):
        path = tmp_path / matrix_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"group": {"standalone": [job(f"job-{index}")]}}))

    # Stub only the parser, retaining the action's directory selection and merger.
    parser = tmp_path / "parser.py"
    parser.write_text(
        "import os, shutil, sys\n"
        "from pathlib import Path\n"
        "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "shutil.copyfile(sys.argv[1], output / 'workflow.json')\n"
        "with open(os.environ['PARSER_LOG'], 'a') as stream:\n"
        "    stream.write(str(output.resolve()) + '\\n')\n"
    )
    action_dir = Path(__file__).resolve().parent
    action = yaml.safe_load((action_dir / "action.yml").read_text())
    step = next(
        step for step in action["runs"]["steps"] if step.get("id") == "build-workflow"
    )
    script = step["run"].split("\nif [[ -f workflow/override.json ]]")[0]
    script = re.sub(
        r"\$\{\{\s*env\.([a-z_]+)\s*\}\}",
        lambda match: (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(parser))}"
            if match[1] == "matrix_parser"
            else ""
        ),
        script,
    )
    log = tmp_path / "parser.log"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env=os.environ
        | {
            "GITHUB_ACTION_PATH": str(action_dir),
            "INPUT_MATRIX_FILES": " ".join(matrix_files),
            "INPUT_WORKFLOWS": "nightly",
            "PARSER_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    directories = log.read_text().splitlines()
    assert len(set(directories)) == len(matrix_files)
    assert Path(directories[0]) == (tmp_path / matrix_files[0]).parent.resolve()
    entries = json.loads((tmp_path / "workflow/workflow.json").read_text())["group"][
        "standalone"
    ]
    assert {entry["name"] for entry in entries} == {
        f"job-{index}" for index in range(len(matrix_files))
    }

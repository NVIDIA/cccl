# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise standalone artifact routing and release checks."""

import hashlib
import io
import json
import runpy
import shlex
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _generate(tmp_path, full=(), lite=(), workflow="pull_request"):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/actions/workflow-build/build-workflow.py"),
            str(ROOT / "ci/matrix.yaml"),
            "--workflows",
            workflow,
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
    return json.loads((tmp_path / "workflow/workflow.json").read_text())


@pytest.mark.parametrize(
    "schedule,lite,suite_args,test_name,gpus",
    [
        ("pull_request", False, "-suite pr", "Test cuda.coop.cutlass", ["l4"]),
        ("pull_request", True, "-suite smoke", "Test cuda.coop.cutlass smoke", ["l4"]),
        ("weekly", False, "", "Test cuda.coop.cutlass full", ["h100", "l4"]),
    ],
)
def test_runtime_uses_the_cpu_qualified_artifact(
    tmp_path, schedule, lite, suite_args, test_name, gpus
):
    projects = ["cuda_coop_cutlass"]
    workflow = _generate(
        tmp_path, () if lite else projects, projects if lite else (), workflow=schedule
    )
    pairs = [p for g in workflow.values() for p in g.get("two_stage", [])]
    assert len(pairs) == 1
    (producer,) = pairs[0]["producers"]
    consumers = pairs[0]["consumers"]
    python_args = ["-py-version", "3.12"]
    assert shlex.split(producer["command"])[1:] == ["-stage", "compile", *python_args]
    assert producer["gpu_count"] == 0
    assert sorted(job["runner"] for job in consumers) == [
        f"linux-amd64-gpu-{gpu}-latest-1" for gpu in gpus
    ]
    runtime_args = ["-stage", "runtime", *suite_args.split(), *python_args]
    for consumer in consumers:
        assert shlex.split(consumer["command"])[1:] == runtime_args
        assert f"] {test_name}(" in consumer["name"]
        assert producer["cuda"] == consumer["cuda"]
        for field in ("py_version", "cpu", "project"):
            assert (
                producer["origin"]["matrix_job"][field]
                == consumer["origin"]["matrix_job"][field]
            )
    jobs = [producer, *consumers] + [
        j for g in workflow.values() for j in g.get("standalone", [])
    ]
    for job in jobs:
        assert (ROOT / job["command"].split('"')[1]).is_file()
        assert "numba" not in job["command"]
        assert "coexistence" not in job["command"]
        if not job["gpu_count"]:
            assert "] Build cuda.coop.cutlass(" in job["name"]
    assert sum(job["gpu_count"] for job in jobs) == len(gpus)


@pytest.mark.parametrize(
    "path",
    [
        "python/cuda_coop_cutlass/cuda/coop/cutlass/_compiler/_runtime.py",
        "python/cuda_coop_cutlass/pyproject.toml",
        "ci/test/test_cuda_coop_cutlass_ci.py",
        "ci/cuda_coop_cutlass_requirements.txt",
        "ci/test_cuda_coop_cutlass.sh",
        "ci/validate_cuda_coop_cutlass_artifacts.py",
        "ci/verify_cuda_coop_cutlass_index.py",
    ],
)
def test_standalone_changes_select_only_standalone_project(path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "ci/inspect_changes.py"), "--stdin"],
        input=path + "\n",
        check=True,
        capture_output=True,
        text=True,
    )
    selection = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("FULL_BUILD=", "LITE_BUILD="))
    )
    assert selection == {"FULL_BUILD": "cuda_coop_cutlass", "LITE_BUILD": ""}


def test_generic_wheel_workflow_excludes_standalone_package(tmp_path):
    workflow = _generate(tmp_path, ["cuda_coop_cutlass"], workflow="python-wheels")
    assert not workflow


@pytest.mark.parametrize(
    "remote_wheel,remote_sdist,allow_missing,error",
    [
        (b"wrong", None, True, "SHA-256 mismatch"),
        (None, None, False, "Missing from test.pypi.org"),
        (None, None, True, None),
        (b"wheel", None, False, None),
        (b"wheel", b"sdist", True, "Unexpected file"),
    ],
)
def test_release_index_hashes(
    monkeypatch, remote_wheel, remote_sdist, allow_missing, error
):
    wheel = "cuda_coop_cutlass-0.1.0-py3-none-any.whl"
    sdist = "cuda_coop_cutlass-0.1.0.tar.gz"
    files = [
        {"filename": name, "digests": {"sha256": hashlib.sha256(data).hexdigest()}}
        for name, data in ((wheel, remote_wheel), (sdist, remote_sdist))
        if data is not None
    ]

    def index_response(url, timeout):
        assert url == "https://test.pypi.org/pypi/cuda-coop-cutlass/0.1.0/json"
        return io.BytesIO(json.dumps({"urls": files}).encode())

    checker = str(ROOT / "ci/verify_cuda_coop_cutlass_index.py")
    flags = ["--allow-missing"] if allow_missing else []
    digest = hashlib.sha256(b"wheel").hexdigest()
    monkeypatch.setattr(sys, "argv", [checker, "testpypi", "0.1.0", digest, *flags])
    monkeypatch.setattr("urllib.request.urlopen", index_response)
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    with pytest.raises(SystemExit, match=error) if error else nullcontext():
        runpy.run_path(checker, run_name="__main__")
    assert sleeps == ([10] * 5 if remote_wheel is None and not allow_missing else [])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check workflow identifiers when native jq emits Windows line endings."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
JQ = shutil.which("jq")
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or BASH is None or JQ is None,
    reason="The native-jq simulation requires Linux, Bash, and jq.",
)


@pytest.mark.parametrize(
    ("compiler", "project", "kind", "expected"),
    [
        ("MSVC", "python", "cccl", "wheel-cccl-windows-amd64-py3.14t"),
        ("MSVC", "python_v2", "cccl", "wheel-cccl-v2-windows-amd64-py3.14t"),
        ("MSVC", "cuda_coop_python", "coop", "wheel-coop-windows-amd64-py3.14t"),
        ("GCC", "python", "cccl", "wheel-cccl-linux-amd64-py3.14t"),
    ],
)
def test_native_jq_artifact_names(tmp_path, compiler, project, kind, expected):
    tools = tmp_path / "native tools"
    tools.mkdir()
    jq = tools / "jq"
    jq.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\n"
        f"result = subprocess.run([{JQ!r}, *sys.argv[1:]], stdout=subprocess.PIPE)\n"
        "sys.stdout.buffer.write(result.stdout.replace(b'\\n', b'\\r\\n'))\n"
        "sys.exit(result.returncode)\n"
    )
    jq.chmod(0o755)
    workflow_dir = tmp_path / "workflow"
    workflow_dir.mkdir()
    (workflow_dir / "workflow.json").write_text(
        json.dumps(
            {
                "Python": {
                    "standalone": [
                        {
                            "id": "E",
                            "origin": {
                                "matrix_job": {
                                    "py_version": "3.14t",
                                    "cxx_family": compiler,
                                    "cpu": "amd64",
                                    "project": project,
                                }
                            },
                        }
                    ]
                }
            }
        )
    )
    env = os.environ | {
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_ACTIONS": "true",
        "RUNNER_TEMP": str(tmp_path),
        "JOB_ID": "E",
        "CCCL_WHEEL_KIND": kind,
    }
    for script, value in (
        ("get_wheel_artifact_name.sh", expected),
        ("get_job_project.sh", project),
    ):
        result = subprocess.run(
            [BASH, str(REPO_ROOT / "ci/util/workflow" / script)],
            env=env,
            capture_output=True,
            check=True,
        )
        assert result.stdout == f"{value}\n".encode()

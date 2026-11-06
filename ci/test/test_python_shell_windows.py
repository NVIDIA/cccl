# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for native Windows Python output and environment paths."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or BASH is None,
    reason="These native-tool simulations use Linux executable semantics.",
)


@pytest.fixture
def run_python_helper(tmp_path):
    repo = tmp_path / "repository with spaces"
    ci = repo / "ci"
    ci.mkdir(parents=True)
    for name in ("pyenv_helper.sh", "pretty_printing.sh"):
        shutil.copy2(REPO_ROOT / "ci" / name, ci / name)
    tools = tmp_path / "tools"
    tools.mkdir()
    scripts = {
        "python": 'printf "%s\\r\\n" "${MOCK_PROBE:-FT=1}"; exit "${MOCK_EXIT:-0}"',
        "cygpath": 'printf "C:/Native Paths/%s\\n" "${2##*/}"',
        "uv": 'printf "%s\\n" "$@" > "$MOCK_UV_ARGS"',
        "nvcc": 'printf "Cuda compilation tools, release 13.4, V13.4.92\\r\\n"',
    }
    for name, script in scripts.items():
        path = tools / name
        path.write_text(f"#!/usr/bin/env bash\n{script}\n")
        path.chmod(0o755)

    def run(script, extra_env=None):
        env = os.environ.copy()
        env.pop("BASH_ENV", None)
        env.update(
            PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
            TMPDIR=str(tmp_path),
            MOCK_UV_ARGS=str(tmp_path / "uv-args.txt"),
            CCCL_CUDA_VERSION="13.4",
        )
        env.update(extra_env or {})
        result = subprocess.run(
            [
                BASH,
                "--noprofile",
                "--norc",
                "-c",
                "set -euo pipefail; OSTYPE=msys; source ci/pyenv_helper.sh; " + script,
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result, repo

    return run


@pytest.mark.parametrize("probe,status", [("FT=1", 0), ("FT=0", 1)])
def test_free_threading_probe_accepts_native_crlf(run_python_helper, probe, status):
    result, _ = run_python_helper(
        "is_free_threaded_python", extra_env={"MOCK_PROBE": probe}
    )
    assert result.returncode == status, result.stdout + result.stderr
    assert "ERROR" not in result.stderr


def test_free_threading_probe_does_not_mask_native_failure(run_python_helper):
    result, _ = run_python_helper(
        "is_free_threaded_python", extra_env={"MOCK_EXIT": "7"}
    )
    assert result.returncode != 0
    assert "probe exited 7" in result.stderr


def test_pip_constraint_is_written_then_exported_as_native_path(run_python_helper):
    result, repo = run_python_helper(
        'pin_cuda_toolkit pinned; printf "%s\\n" "$PIP_CONSTRAINT"'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "C:/Native Paths/ctk-constraint.txt"
    assert (repo.parent / "ctk-constraint.txt").read_text() == "cuda-toolkit==13.4.*\n"


def test_windows_venv_exports_native_path_without_activation(run_python_helper):
    result, repo = run_python_helper(
        'setup_python_env 3.14t .cccl-test-venv; printf "VENV=%s\\n" "$VIRTUAL_ENV"'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VENV=C:/Native Paths/.cccl-test-venv" in result.stdout
    args = (repo.parent / "uv-args.txt").read_text().splitlines()
    assert args[:4] == ["venv", "--seed", "--python", "3.14t"]

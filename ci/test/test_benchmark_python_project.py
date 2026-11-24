# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise benchmark project routing without installing packages or using a GPU."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ci/bench/compare_paths.sh"


def _executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def _run(tmp_path, project=None, pattern=r".*\.py", cuda_major=13):
    source = tmp_path / "source with spaces"
    source.mkdir()
    (source / "cccl-version.json").write_text("{}")
    selected = project or "cuda_cccl"
    benchmarks = source / "python" / selected / "benchmarks"
    benchmarks.mkdir(parents=True)
    target = benchmarks / "run.py"
    target.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['RUN_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv) + '\\n')\n"
        "for flag in ('--jsonbin', '--md'):\n"
        "    Path(sys.argv[sys.argv.index(flag) + 1]).write_text('{}')\n"
    )
    (benchmarks / "_cases.py").write_text("raise AssertionError('helper ran')\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    _executable(
        tools / "python3",
        "#!/bin/bash\n"
        "if [[ $1 == -m && $2 == venv ]]; then\n"
        '  mkdir -p "$3/bin"\n'
        f'  ln -s {shlex.quote(sys.executable)} "$3/bin/python"\n'
        '  cp "$FAKE_PIP" "$3/bin/pip"\n'
        "else\n"
        f'  exec {shlex.quote(sys.executable)} "$@"\n'
        "fi\n",
    )
    _executable(
        tools / "pip",
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$PIP_LOG"\n',
    )
    _executable(
        tools / "nvcc",
        f'#!/bin/bash\necho "Cuda compilation tools, release {cuda_major}.3"\n',
    )
    _executable(tools / "compare", '#!/bin/bash\nprintf "# Summary\\n\\nFixture\\n"\n')
    env = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ["PATH"],
        FAKE_PIP=str(tools / "pip"),
        PIP_LOG=str(tmp_path / "pip.log"),
        RUN_LOG=str(tmp_path / "runs.jsonl"),
        CCCL_BENCH_ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        CCCL_BENCH_BUILD_ROOT=str(tmp_path / "build"),
        CCCL_BENCH_COMPARE_BIN=str(tools / "compare"),
        CCCL_BENCH_LEGACY_COMPARE_BIN=str(tools / "compare"),
    )
    command = ["bash", str(SCRIPT), str(source), str(source)]
    if project is not None:
        command += ["--python-project", project]
    command += ["--python-filter", pattern, "--nvbench-args", "--profile"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    return result, target


@pytest.mark.parametrize("project", [None, "cuda_coop_cutlass"])
def test_selected_project_installs_and_runs_in_both_environments(tmp_path, project):
    result, target = _run(tmp_path, project)
    assert result.returncode == 0, result.stdout + result.stderr
    installs = (tmp_path / "pip.log").read_text()
    extra = "bench" if project else "bench-cu13"
    assert installs.count(f"install -e {target.parents[1]}[{extra}]") == 2
    runs = [
        json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()
    ]
    assert len(runs) == 2
    for args in runs:
        assert args[0] == str(target)
        assert "--profile" in args
        assert "--jsonbin" in args and "--md" in args


def test_no_matching_targets_does_not_install_dependencies(tmp_path):
    result, _ = _run(tmp_path, "cuda_coop_cutlass", pattern="missing")
    assert result.returncode != 0
    assert "No Python benchmark scripts matched" in result.stderr
    assert not (tmp_path / "pip.log").exists()


def test_cutlass_rejects_cuda12_before_installing(tmp_path):
    result, _ = _run(tmp_path, "cuda_coop_cutlass", cuda_major=12)
    assert result.returncode != 0
    assert "require CUDA 13" in result.stderr
    assert not (tmp_path / "pip.log").exists()


def test_unknown_project_is_rejected(tmp_path):
    result, _ = _run(tmp_path, "unknown")
    assert result.returncode != 0
    assert "Unsupported Python project" in result.stderr
    assert not (tmp_path / "pip.log").exists()

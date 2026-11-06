# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the Windows sibling-container contract without Docker or Windows.

Native Git Bash relocation and Windows process startup still require CI.
"""

import json
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
    reason="The native-path simulation uses Linux executable semantics.",
)

MOCK_TOOL = r"""
import json
import os
import shutil
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "cygpath":
    print(args[-1].replace("\\", "/"))
elif name == "nvcc":
    sys.stdout.buffer.write(b"Cuda compilation tools, release 13.4, V13.4.92\r\n")
elif name == "cp":
    source = Path(os.environ["MOCK_GIT"]) / args[-2].lstrip("/")
    shutil.copytree(source, Path(args[-1]) / source.name)
elif name == "docker":
    if args[0] == "version":
        sys.exit(0)
    runtime, = Path(os.environ["MOCK_REPO"]).glob(".cccl-minimal-bash.*")
    record = {
        "args": args,
        "no_pathconv": os.environ.get("MSYS_NO_PATHCONV"),
        "arg_excl": os.environ.get("MSYS2_ARG_CONV_EXCL"),
        "staged_files": sorted(str(p.relative_to(runtime)) for p in runtime.rglob("*") if p.is_file()),
        "bootstrap": (runtime / "run.sh").read_text(),
    }
    Path(os.environ["MOCK_LOG"]).write_text(json.dumps(record))
    sys.exit(int(os.environ.get("MOCK_DOCKER_EXIT", "0")))
"""


@pytest.fixture
def run_container(tmp_path):
    repo = tmp_path / "repository with spaces"
    script = repo / "ci/util/python/run_in_minimal_container.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "ci/util/python/run_in_minimal_container.sh", script)

    git = tmp_path / "Git runtime"
    for filename in (
        "usr/bin/bash.exe",
        "usr/bin/msys-2.0.dll",
        "usr/bin/cygpath.exe",
        "usr/bin/link.exe",
        "mingw64/bin/curl.exe",
        "etc/ssl/certs/ca-bundle.crt",
        "etc/fstab",
        "etc/profile",
    ):
        path = git / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")

    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("cygpath", "nvcc", "cp", "docker"):
        path = tools / name
        path.write_text(f"#!{sys.executable}\n{MOCK_TOOL}")
        path.chmod(0o755)

    def run(extra_env=None, bundled_dll=None):
        if bundled_dll:
            (git / "usr/bin" / bundled_dll).write_text("must not enter container\n")
        log = tmp_path / "docker.json"
        env = os.environ.copy()
        for key in ("BASH_ENV", "CCCL_CUDA_VERSION", "CCCL_MINIMAL_CONTAINER_IMAGE"):
            env.pop(key, None)
        env.update(
            {
                "OSTYPE": "msys",
                "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                "HOST_WORKSPACE": "C:/host workspace/cccl",
                "CONTAINER_WORKSPACE": "C:/container workspace/cccl",
                "MOCK_GIT": str(git),
                "MOCK_REPO": str(repo),
                "MOCK_LOG": str(log),
            }
        )
        env.update(extra_env or {})
        result = subprocess.run(
            [
                BASH,
                "--noprofile",
                "--norc",
                str(script),
                "ci/util/python/run_headers_tests.sh",
                "-py-version",
                "3.14t",
                "argument with spaces",
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert not list(repo.glob(".cccl-minimal-bash.*")), "Bash staging leaked"
        return result, json.loads(log.read_text()) if log.exists() else None

    return run


@pytest.mark.parametrize("needs_gpu", [False, True])
def test_windows_native_mounts_arguments_and_device_selection(run_container, needs_gpu):
    result, record = run_container(
        {"CCCL_MINIMAL_CONTAINER_NO_GPU": "0" if needs_gpu else "1"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = record["args"]
    assert "--isolation=process" in args
    assert "--gpus" not in args
    assert ("--device" in args) == needs_gpu
    if needs_gpu:
        assert args[args.index("--device") + 1] == (
            "class/5B45201D-F2F2-4F3B-85BB-30FF1F953599"
        )
    assert args[args.index("--mount") + 1] == (
        "type=bind,source=C:/host workspace/cccl,target=C:/container workspace/cccl"
    )
    assert "CCCL_CUDA_VERSION=13.4" in args
    assert "CCCL_INSIDE_MINIMAL_CONTAINER=1" in args
    assert args[-4:] == [
        "ci/util/python/run_headers_tests.sh",
        "-py-version",
        "3.14t",
        "argument with spaces",
    ]
    assert any(arg.endswith("/usr/bin/bash.exe") for arg in args)
    assert record["no_pathconv"] == "1"
    assert record["arg_excl"] == "*"
    assert "usr/bin/msys-2.0.dll" in record["staged_files"]
    assert "usr/bin/link.exe" in record["staged_files"]
    assert "etc/fstab" not in record["staged_files"]
    assert "etc/profile" not in record["staged_files"]
    subprocess.run([BASH, "-n"], input=record["bootstrap"], text=True, check=True)


def test_container_failure_propagates_and_cleans_staging(run_container):
    result, record = run_container({"MOCK_DOCKER_EXIT": "42"})
    assert record is not None
    assert result.returncode == 42, result.stdout + result.stderr


@pytest.mark.parametrize("dll", ["msvcp140.dll", "VCRUNTIME140_1.DLL"])
def test_bundled_msvc_runtime_is_rejected_before_starting_container(run_container, dll):
    result, record = run_container(bundled_dll=dll)
    assert result.returncode != 0
    assert record is None
    assert "Git Bash staging would supply an MSVC runtime" in result.stderr

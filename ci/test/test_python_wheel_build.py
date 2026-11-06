# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise wheel orchestration without Docker, CUDA, or package downloads."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def wheel_build(tmp_path):
    repo = tmp_path / "checkout with spaces"
    ci = repo / "ci"
    ci.mkdir(parents=True)
    (ci / "util/python").mkdir(parents=True)
    (repo / "python/cuda_cccl").mkdir(parents=True)
    for name in ["build_cuda_cccl_python.sh", "build_cuda_cccl_wheel.sh"]:
        shutil.copy(REPO_ROOT / "ci" / name, ci / name)
    shutil.copy(
        REPO_ROOT / "ci/util/python/common_arg_parser.sh",
        ci / "util/python/common_arg_parser.sh",
    )
    (ci / "pyenv_helper.sh").write_text(
        'cccl_python_is_windows() { [[ "$MOCK_WINDOWS" == 1 ]]; }\n'
        "setup_python_env() { :; }\n"
    )
    tools = tmp_path / "mock tools"
    tools.mkdir()
    mock = tools / "mock"
    mock.write_text(
        f"#!{sys.executable}\n"
        r"""
import json
import os
import subprocess
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
repo = Path(os.environ["MOCK_REPO"])
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"tool": name, "args": args, "env": {
        key: os.environ.get(key) for key in [
            "CC", "CXX", "CUDACXX", "CUDAHOSTCXX", "CMAKE_GENERATOR",
            "MSYS_NO_PATHCONV", "MSYS2_ARG_CONV_EXCL",
        ]
    }}) + "\n")

def wheel(directory, suffix="", content="built"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"cuda_cccl-1.0-py3-none-win_amd64{suffix}.whl").write_text(content)

failure = os.environ.get("MOCK_FAILURE")
if name == "cygpath":
    path = args[-1].replace("\\", "/")
    if args[0] == "-u":
        print(path[2:] if path[1:2] == ":" else path)
    else:
        print(path if path[1:2] == ":" else "C:" + path)
elif name == "nvcc":
    major = "13" if "cuda13" in sys.argv[0] else "12"
    print(f"Cuda compilation tools, release {major}.9, V{major}.9.1\r")
elif name == "grep":
    perl_regexp = any(
        arg == "--perl-regexp" or (arg.startswith("-") and "P" in arg)
        for arg in args
    )
    if os.environ.get("MOCK_PCRE_SUPPORTED") == "0" and perl_regexp:
        print("grep: -P supports only unibyte and UTF-8 locales", file=sys.stderr)
        sys.exit(2)
    os.execv(os.environ["MOCK_REAL_GREP"], ["grep", *args])
elif name == "docker":
    if args[0] != "run":
        sys.exit(0)
    if failure == "docker":
        sys.exit(7)
    if os.environ["MOCK_WINDOWS"] == "1":
        env = os.environ.copy()
        for index, arg in enumerate(args):
            if arg == "--env":
                key, value = args[index + 1].split("=", 1)
                env[key] = value
        sys.exit(subprocess.run(
            ["bash", str(repo / "ci/build_cuda_cccl_wheel.sh")],
            env=env, check=False,
        ).returncode)
    major = "13" if any("-cuda13" in arg for arg in args) else "12"
    wheel(repo / "wheelhouse", f".cu{major}")
elif name == "python":
    if args == ["-"]:
        sys.stdin.read()
        print("28\r")
    elif args[:3] == ["-m", "pip", "wheel"]:
        if failure == "pip":
            sys.exit(9)
        if failure != "missing_build_wheel":
            wheel(Path.cwd() / "dist")
        if failure == "duplicate_build_wheels":
            (Path.cwd() / "dist/cuda_cccl-2.0-py3-none-win_amd64.whl").touch()
    elif args and args[0].endswith("merge_cuda_wheels.py"):
        assert all(Path(path).is_file() for path in args[1:3])
        wheel(repo / "wheelhouse_merged", content="merged")
    elif len(args) > 2 and args[1] in ["delvewheel", "auditwheel"]:
        if failure == "repair":
            sys.exit(11)
        if failure != "missing_repaired_wheel":
            wheel(repo / "wheelhouse_final", content="repaired")
"""
    )
    mock.chmod(0o755)
    for name in ["python", "docker", "cygpath", "cl", "grep"]:
        (tools / name).symlink_to(mock)
    env = os.environ.copy()
    env.update(
        PATH=f"{tools}:{env['PATH']}",
        MOCK_LOG=str(tmp_path / "calls.jsonl"),
        MOCK_REPO=str(repo),
        MOCK_REAL_GREP=shutil.which("grep"),
        HOST_WORKSPACE="D:/runner workspace",
        CONTAINER_WORKSPACE=f"C:{repo}",
        NUMBER_OF_PROCESSORS="16",
        SystemRoot=f"C:{tmp_path}/Windows",
        GITHUB_ACTIONS="",
    )
    system32 = tmp_path / "Windows/System32"
    system32.mkdir(parents=True)
    (system32 / "msvcp140.dll").touch()
    for version in ["12_8", "12_9", "13_0"]:
        toolkit = tmp_path / f"cuda{version.replace('_', '.')} toolkit"
        (toolkit / "bin").mkdir(parents=True)
        (toolkit / "bin/nvcc").symlink_to(mock)
        (toolkit / "bin/nvcc.exe").symlink_to(mock)
        env[f"CUDA_PATH_V{version}"] = f"C:{toolkit}"

    def run(windows=True, failure="", stale=False, pcre_supported=True):
        if stale:
            for relative_path in [
                "wheelhouse/cuda_cccl-0.0-py3-none-win_amd64.cu12.whl",
                "wheelhouse/cuda_cccl-0.0-py3-none-win_amd64.cu13.whl",
                "python/cuda_cccl/dist/cuda_cccl-0.0-py3-none-win_amd64.whl",
                "wheelhouse_merged/cuda_cccl-0.0-py3-none-win_amd64.whl",
                "wheelhouse_final/cuda_cccl-0.0-py3-none-win_amd64.whl",
            ]:
                stale_path = repo / relative_path
                stale_path.parent.mkdir(parents=True, exist_ok=True)
                stale_path.write_text("stale")
        result = subprocess.run(
            ["bash", str(ci / "build_cuda_cccl_python.sh"), "-py-version", "3.13"],
            cwd=repo,
            env=env
            | {
                "MOCK_WINDOWS": str(int(windows)),
                "MOCK_FAILURE": failure,
                "MOCK_PCRE_SUPPORTED": str(int(pcre_supported)),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        calls = [
            json.loads(line) for line in Path(env["MOCK_LOG"]).read_text().splitlines()
        ]
        return result, calls, repo

    return run


@pytest.mark.parametrize("stale", [False, True])
def test_windows_builds_both_native_wheels_then_repairs(wheel_build, stale):
    result, calls, repo = wheel_build(stale=stale)
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [call for call in calls if call["args"][:3] == ["-m", "pip", "wheel"]]
    assert len(builds) == 2
    for build, version in zip(builds, ["12.9", "13.0"]):
        assert build["env"]["CXX"] == "cl.exe"
        assert build["env"]["CUDAHOSTCXX"] == "cl.exe"
        assert build["env"]["CMAKE_GENERATOR"] == "Ninja"
        assert f"cuda{version} toolkit/bin/nvcc.exe" in build["env"]["CUDACXX"]
        assert any(
            arg.startswith("cmake.define.CUDAToolkit_ROOT=C:")
            and f"cuda{version} toolkit" in arg
            for arg in build["args"]
        )
    (nested,) = [
        call for call in calls if call["tool"] == "docker" and call["args"][0] == "run"
    ]
    assert nested["env"]["MSYS2_ARG_CONV_EXCL"] == "*"
    assert f"type=bind,source=D:/runner workspace,target=C:{repo}" in nested["args"]
    assert nested["args"][-4:] == [
        "C:/Program Files/Git/bin/bash.exe",
        "--noprofile",
        "--norc",
        f"C:{repo}/ci/build_cuda_cccl_wheel.sh",
    ]
    (repair,) = [
        call for call in calls if call["args"][:3] == ["-m", "delvewheel", "repair"]
    ]
    assert "--analyze-existing" in repair["args"]
    assert repair["args"][repair["args"].index("--namespace-pkg") + 1] == "cuda"
    assert "dbghelp.dll" in repair["args"][repair["args"].index("--exclude") + 1]
    assert repair["env"]["MSYS2_ARG_CONV_EXCL"] is None
    (final,) = (repo / "wheelhouse").glob("*.whl")
    assert final.read_text() == "repaired"
    assert ".cu12" not in final.name and ".cu13" not in final.name


def test_linux_keeps_manylinux_builds_and_auditwheel(wheel_build):
    result, calls, repo = wheel_build(windows=False)
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [
        call for call in calls if call["tool"] == "docker" and call["args"][0] == "run"
    ]
    assert len(builds) == 2
    for build, version in zip(builds, ["12.9.1", "13.1.1"]):
        assert any(
            arg.startswith(f"rapidsai/ci-wheel:26.04-cuda{version}")
            for arg in build["args"]
        )
        assert build["args"][-1] == "/workspace/ci/build_cuda_cccl_wheel.sh"
        assert "CCACHE_DIR=/root/.ccache" in build["args"]
        assert "CPM_SOURCE_CACHE=/root/.cpm-cache" in build["args"]
    assert any(call["args"][:3] == ["-m", "auditwheel", "repair"] for call in calls)
    assert not any("delvewheel" in call["args"] for call in calls)
    (final,) = (repo / "wheelhouse").glob("*.whl")
    assert final.read_text() == "repaired"


def test_windows_cuda_version_parsing_without_pcre_locale(wheel_build):
    result, _, _ = wheel_build(pcre_supported=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Detected CUDA version: 12\n" in result.stdout
    assert "Detected CUDA version: 13\n" in result.stdout


@pytest.mark.parametrize(
    "failure,exit_code",
    [("pip", 9), ("docker", 7), ("repair", 11), ("missing_repaired_wheel", 1)],
)
def test_windows_wheel_failures_propagate(wheel_build, failure, exit_code):
    result, calls, _ = wheel_build(failure=failure)
    assert result.returncode == exit_code, result.stdout + result.stderr
    if failure in ["pip", "docker"]:
        assert not any("delvewheel" in call["args"] for call in calls)


@pytest.mark.parametrize(
    "failure",
    ["missing_build_wheel", "duplicate_build_wheels", "missing_repaired_wheel"],
)
def test_stale_or_ambiguous_wheels_cannot_satisfy_success(wheel_build, failure):
    result, calls, repo = wheel_build(failure=failure, stale=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not list((repo / "wheelhouse").glob("cuda_cccl-0.0-*.whl"))
    if failure != "missing_repaired_wheel":
        assert not any("delvewheel" in call["args"] for call in calls)
        assert "expected exactly one built CUDA 12 wheel" in result.stderr

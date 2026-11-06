# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shared build driver with native Windows tools simulated on Linux.

Run with: python3 -m pytest ci/test/test_build_common_windows.py
The real Windows CI experiment covers Git Bash and native process interoperability.
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
    reason="The native-path simulation uses Linux filesystem and executable semantics.",
)

MOCK_TOOL = r"""
import json
import os
import subprocess
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({
        "tool": name,
        "args": args,
        "env": {key: os.environ.get(key) for key in (
            "CXX", "CUDACXX", "CUDAHOSTCXX", "CMAKE_BUILD_PARALLEL_LEVEL",
        )},
    }) + "\n")

if name == "cygpath":
    path = args[1].replace("\\", "/")
    if args[0] == "-u":
        print(str(Path(os.environ["MOCK_TOOLS"]) / path.rsplit("/", 1)[1])
              if path.startswith("C:/") else path)
    elif args[0] == "-am":
        print("C:/Native Tools/" + Path(path).stem + ".exe")
    else:
        sys.exit(99)
elif name in ("nvcc", "nvcc.exe"):
    sys.stdout.buffer.write(b"Cuda compilation tools, release 13.4, V13.4.70\r\n")
elif name == "nproc":
    print("7")
elif name == "jq":
    sys.stdout.write("thrust" if "-j" in args else "thrust\r\n")
elif name == "timeout":
    sys.exit(subprocess.call(args[1:]))
elif name == "memmon.sh" and args[0] == "--start":
    Path(args[args.index("--log-file") + 1]).write_text("mock memory log\n")
elif name == "cmake" and "--build" in args:
    sys.exit(int(os.environ.get("MOCK_BUILD_EXIT", "0")))
elif name == "sccache":
    print("{}")
"""


@pytest.fixture
def run_common(tmp_path):
    repo = tmp_path / "repository with spaces"
    ci = repo / "ci"
    (ci / "util").mkdir(parents=True)
    for source in (
        "build_common.sh",
        "build_cub.sh",
        "build_cudax.sh",
        "pretty_printing.sh",
        "util/retry.sh",
        "util/extract_switches.sh",
        "util/version_compare.sh",
    ):
        shutil.copy2(REPO_ROOT / "ci" / source, ci / source)
    (repo / "CMakePresets.json").write_text("{}")

    tools = tmp_path / "tools with spaces"
    tools.mkdir()
    tool_source = f"#!{sys.executable}\n{MOCK_TOOL}"
    for name in (
        "cl",
        "cl.exe",
        "g++",
        "nvcc",
        "nvcc.exe",
        "cygpath",
        "cmake",
        "git",
        "jq",
        "ninja",
        "nvidia-smi",
        "nproc",
        "sccache",
        "timeout",
        "probe",
    ):
        path = tools / name
        path.write_text(tool_source)
        path.chmod(0o755)
    for name in ("util/memmon.sh", "ninja_summary.py"):
        path = ci / name
        path.write_text(tool_source)
        path.chmod(0o755)

    # Linux treats a native C:/ path as relative. This alias lets the real Bash
    # driver execute its normalized nvcc path without patching the source.
    (ci / "C:").mkdir()
    (ci / "C:" / "Native Tools").symlink_to(tools, target_is_directory=True)

    def run(platform="msys", extra_env=None, args=(), script=None):
        env = os.environ.copy()
        for key in (
            "CXX",
            "CUDACXX",
            "PARALLEL_LEVEL",
            "PEDANTIC",
            "VERBOSE",
            "MEMMON",
            "BASH_ENV",
        ):
            env.pop(key, None)
        env.update(
            {
                "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                "MOCK_TOOLS": str(tools),
                "MOCK_LOG": str(tmp_path / "tools.jsonl"),
                "MOCK_OSTYPE": platform,
                "GITHUB_ACTIONS": "true",
                "CCCL_BUILD_INFIX": "probe",
                "CCCL_CI_COMMAND_TIMEOUT": "5.5h",
                "NUMBER_OF_PROCESSORS": "16",
            }
        )
        env.update(extra_env or {})
        result = subprocess.run(
            [
                BASH,
                "--noprofile",
                "--norc",
                "-c",
                """
set -euo pipefail
OSTYPE="$MOCK_OSTYPE"
"""
                + (
                    script
                    or """
source ./ci/build_common.sh
configure_and_build_preset Probe thrust "-DCMAKE_CXX_STANDARD=$CXX_STANDARD"
probe "$NVCC_VERSION" "$N_CPUS" "$BUILD_DIR"
"""
                ),
                "build-common-test",
                *args,
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        records = [
            json.loads(line)
            for line in (tmp_path / "tools.jsonl").read_text().splitlines()
        ]
        return result, records, repo

    return run


def test_native_compiler_paths_crlf_and_architecture_list(run_common):
    result, records, repo = run_common(
        extra_env={
            "CXX": r"C:\Native Tools\cl.exe",
            "CUDACXX": r"C:\Native Tools\nvcc.exe",
            "MEMMON": "1",
        },
        args=("-arch", "75;89-virtual", "-std", "20"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    configure, build = [item for item in records if item["tool"] == "cmake"]
    for invocation in (configure, build):
        assert invocation["env"] == {
            "CXX": "C:/Native Tools/cl.exe",
            "CUDACXX": "C:/Native Tools/nvcc.exe",
            "CUDAHOSTCXX": "C:/Native Tools/cl.exe",
            "CMAKE_BUILD_PARALLEL_LEVEL": "16",
        }
    assert "-DCMAKE_CUDA_ARCHITECTURES=75;89-virtual" in configure["args"]
    assert "-DCMAKE_CXX_STANDARD=20" in configure["args"]
    probe = next(item for item in records if item["tool"] == "probe")
    assert probe["args"] == ["13.4.70", "16", str(repo / "build/probe")]
    assert not {"nproc", "timeout", "memmon.sh"}.intersection(
        item["tool"] for item in records
    )
    assert (repo / "build/probe/thrust").is_dir()
    assert not (repo / "build/latest").exists()
    assert not (repo / "build/preset-latest").exists()


def test_windows_parallel_override_and_default_host_compiler(run_common):
    result, records, _ = run_common(extra_env={"PARALLEL_LEVEL": "3"})
    assert result.returncode == 0, result.stdout + result.stderr
    build = next(
        item
        for item in records
        if item["tool"] == "cmake" and "--build" in item["args"]
    )
    assert build["env"]["CXX"] == "C:/Native Tools/cl.exe"
    assert build["env"]["CMAKE_BUILD_PARALLEL_LEVEL"] == "3"
    assert build["args"][build["args"].index("--parallel") + 1] == "3"


@pytest.mark.parametrize("cpu_count", ["", "0", "invalid"])
def test_invalid_windows_processor_count_fails_before_configure(run_common, cpu_count):
    result, records, _ = run_common(extra_env={"NUMBER_OF_PROCESSORS": cpu_count})
    assert result.returncode != 0
    assert "NUMBER_OF_PROCESSORS must be a positive integer" in result.stderr
    assert not any(item["tool"] == "cmake" for item in records)


def test_windows_build_failure_survives_statistics_collection(run_common):
    result, records, _ = run_common(extra_env={"MOCK_BUILD_EXIT": "7"})
    assert result.returncode == 7, result.stdout + result.stderr
    assert any(
        item["tool"] == "sccache" and "--show-adv-stats" in item["args"]
        for item in records
    )
    assert not any(item["tool"] == "probe" for item in records)


def test_linux_keeps_cpu_reservation_symlinks_timeout_and_memory_monitor(run_common):
    result, records, repo = run_common(platform="linux-gnu")
    assert result.returncode == 0, result.stdout + result.stderr
    nproc = next(item for item in records if item["tool"] == "nproc")
    assert nproc["args"] == ["--all", "--ignore=1"]
    configure = next(item for item in records if item["tool"] == "cmake")
    assert configure["env"]["CXX"].endswith("/tools with spaces/g++")
    assert configure["env"]["CMAKE_BUILD_PARALLEL_LEVEL"] == "7"
    assert not any(item["tool"] == "cygpath" for item in records)
    timeout = next(item for item in records if item["tool"] == "timeout")
    assert timeout["args"][:3] == ["5.5h", "cmake", "--build"]
    assert [item["args"][0] for item in records if item["tool"] == "memmon.sh"] == [
        "--start",
        "--stop",
    ]
    assert (repo / "build/latest").is_symlink()
    assert (repo / "build/latest").resolve() == repo / "build/probe"
    assert (repo / "build/preset-latest").is_symlink()
    assert (repo / "build/preset-latest").resolve() == repo / "build/probe/thrust"


@pytest.mark.parametrize("platform,enabled", [("msys", "false"), ("linux-gnu", "true")])
def test_cub_benchmarks_preserve_platform_defaults(run_common, platform, enabled):
    result, records, _ = run_common(
        platform=platform,
        extra_env={"GITHUB_ACTIONS": ""},
        script="source ./ci/build_cub.sh",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    configure = next(
        item
        for item in records
        if item["tool"] == "cmake" and "--preset=cub" in item["args"]
    )
    assert f"-DCCCL_ENABLE_BENCHMARKS={enabled}" in configure["args"]
    assert "-DCUB_ENABLE_RDC_TESTS=true" in configure["args"]


@pytest.mark.parametrize("platform,enabled", [("msys", "false"), ("linux-gnu", "true")])
def test_cudax_cufile_preserves_platform_defaults(run_common, platform, enabled):
    result, records, _ = run_common(
        platform=platform,
        extra_env={"GITHUB_ACTIONS": ""},
        script="source ./ci/build_cudax.sh",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    configure = next(
        item
        for item in records
        if item["tool"] == "cmake" and "--preset=cudax" in item["args"]
    )
    assert f"-Dcudax_ENABLE_CUFILE={enabled}" in configure["args"]


def test_windows_smoke_uses_native_jq_output_and_executable_suffix(run_common):
    result, records, _ = run_common(
        script="""
source ./ci/build_common.sh
mkdir -p "$BUILD_DIR/thrust/bin"
cp "$MOCK_TOOLS/probe" "$BUILD_DIR/thrust/bin/cccl.test.cuda_runtime_smoke.exe"
run_cuda_smoke_test Probe thrust-gpu
"""
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(item["tool"] == "cccl.test.cuda_runtime_smoke.exe" for item in records)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_target_driver_preserves_native_build_path_without_line_ending(
    tmp_path, newline
):
    (tmp_path / "cccl-version.json").write_text("{}")
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "tools.jsonl"
    tool_source = (
        f"#!{sys.executable}\n"
        + r"""
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"tool": name, "args": args}) + "\n")
if name == "git":
    print("fixture commit")
elif name == "cmake":
    sys.stdout.write("-- Build files have been written to: C:/Build Tree/thrust" + os.environ["MOCK_NEWLINE"])
"""
    )
    for name in ("git", "cmake", "ninja", "ctest"):
        tool = tools / name
        tool.write_text(tool_source)
        tool.chmod(0o755)
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.update(
        PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
        MOCK_LOG=str(log),
        MOCK_NEWLINE=newline,
    )
    result = subprocess.run(
        [
            BASH,
            str(REPO_ROOT / "ci/util/build_and_test_targets.sh"),
            "--preset",
            "thrust",
            "--build-targets",
            "smoke",
            "--ctest-targets",
            "smoke",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    ninja = next(item for item in records if item["tool"] == "ninja")
    ctest = next(item for item in records if item["tool"] == "ctest")
    assert ninja["args"] == ["-C", "C:/Build Tree/thrust", "smoke"]
    assert ctest["args"] == [
        "--test-dir",
        "C:/Build Tree/thrust",
        "-R",
        "smoke",
        "-V",
        "--output-on-failure",
    ]

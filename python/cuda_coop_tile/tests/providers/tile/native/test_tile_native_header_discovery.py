# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .test_implementation_runtime import (
    NATIVE_ROOT,
    _sdk_include,
    _toolkit_nvrtc,
)


def test_installed_provider_owns_headers_below_unrelated_checkout(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("header discovery regression requires a C++ compiler")
    nvrtc_include, nvrtc_library = _toolkit_nvrtc()
    unrelated = tmp_path / "unrelated-checkout"
    for relative in ("cub/cub", "libcudacxx/include", "thrust"):
        (unrelated / relative).mkdir(parents=True)
    (unrelated / "cub/cub/version.cuh").touch()
    prefix = unrelated / "installed"
    bundle = prefix / "include/cccl"
    (bundle / "cub").mkdir(parents=True)
    (bundle / "cub/version.cuh").touch()
    executable = prefix / "lib/cuda-coop-tile/resolve_headers"
    executable.parent.mkdir(parents=True)
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            '-DCUDA_COOP_TILE_INSTALLED_INCLUDE_RELATIVE="../../include/cccl"',
            f"-I{NATIVE_ROOT}",
            f"-I{_sdk_include()}",
            f"-I{nvrtc_include}",
            str(NATIVE_ROOT.parent / "test/support/include_paths.cpp"),
            str(NATIVE_ROOT / "provider/runtime/include_paths.cpp"),
            str(nvrtc_library),
            f"-Wl,-rpath,{nvrtc_library.parent}",
            "-ldl",
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    env = os.environ.copy()
    for name in (
        "CUDA_COOP_CCCL_ROOT",
        "CCCL_ROOT",
        "CCCL_DIR",
        "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS",
        "CUDA_COOP_TILE_INCLUDE_PATHS",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_PATH",
        "CUDA_TOOLKIT_PATH",
    ):
        env.pop(name, None)

    def resolve(program=executable):
        return subprocess.run(
            [str(program)],
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )

    result = resolve()
    assert result.returncode == 0, result.stderr
    paths = [Path(line) for line in result.stdout.splitlines()]
    assert paths[0] == bundle
    assert unrelated / "cub" not in paths

    outside_build = tmp_path / "outside-build/cpp"
    (outside_build / "CMakeFiles/cuda_coop_tile_provider.dir").mkdir(
        parents=True
    )
    (outside_build / "cuda-coop-tile-cccl-root.txt").write_text(
        f"{unrelated}\n"
    )
    outside_program = outside_build / executable.name
    shutil.copy2(executable, outside_program)
    result = resolve(outside_program)
    assert result.returncode == 0, result.stderr
    assert (
        Path(result.stdout.splitlines()[0]) == unrelated / "libcudacxx/include"
    )

    env["CUDA_COOP_CCCL_ROOT"] = str(unrelated)
    result = resolve()
    assert result.returncode == 0, result.stderr
    assert (
        Path(result.stdout.splitlines()[0]) == unrelated / "libcudacxx/include"
    )

    env["CUDA_COOP_CCCL_ROOT"] = str(bundle)
    result = resolve()
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.splitlines()[0]) == bundle

    env["CUDA_COOP_CCCL_ROOT"] = str(tmp_path / "missing")
    env["CCCL_ROOT"] = str(unrelated)
    result = resolve()
    assert result.returncode != 0
    assert "[LP-HEADERS-001]" in result.stderr
    assert "CUDA_COOP_CCCL_ROOT" in result.stderr

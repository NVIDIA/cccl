# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

NATIVE_ROOT = PACKAGE_ROOT.parents[1] / "tile/cpp/tilelib"

CPP_TEST = (
    PACKAGE_ROOT.parents[1] / "tile/cpp/test/support/implementation_runtime.cpp"
)


def _sdk_include() -> Path:
    raw_root = os.environ.get("CUDA_COOP_TILE_SDK_ROOT")
    if not raw_root:
        pytest.skip("host conformance test requires CUDA_COOP_TILE_SDK_ROOT")
    include = Path(raw_root) / "include"
    assert (include / "tile_ir/linkage_provider/LinkageProviderAPI.h").is_file()
    return include


def _toolkit_nvrtc() -> tuple[Path, Path]:
    raw_root = os.environ.get("CUDA_COOP_TILE_CUDA_ROOT")
    if raw_root:
        root = Path(raw_root)
        for include, library in (
            (
                root / "bin" / "nvrtc" / "include",
                root / "bin" / "nvrtc" / "lib64" / "libnvrtc.so",
            ),
            (root / "include", root / "lib64" / "libnvrtc.so"),
        ):
            if include.joinpath("nvrtc.h").is_file() and library.is_file():
                return include, library
    pytest.skip("host conformance test requires CUDA_COOP_TILE_CUDA_ROOT")


def test_implementation_selection_emits_ltoir_and_exact_scratch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(PACKAGE_ROOT.parents[1]))
    if compiler_root := os.environ.get("CUDA_COOP_TILE_CUDA_ROOT"):
        monkeypatch.setenv("CUDA_HOME", compiler_root)
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    sdk_include = _sdk_include()
    nvrtc_include, nvrtc_library = _toolkit_nvrtc()
    executable = tmp_path / "tile_native_provider_frontend_conformance"

    provider_sources = sorted((NATIVE_ROOT / "provider").rglob("*.cpp"))
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-g0",
            "-ffunction-sections",
            "-fdata-sections",
            "-pthread",
            f"-I{NATIVE_ROOT}",
            f"-I{sdk_include}",
            f"-I{nvrtc_include}",
            str(CPP_TEST),
            *(str(source) for source in provider_sources),
            str(nvrtc_library),
            "-ldl",
            f"-Wl,-rpath,{nvrtc_library.parent}",
            "-Wl,--gc-sections",
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, (
        f"host conformance compile failed:\n{compile_result.stdout}"
        f"{compile_result.stderr}"
    )

    run_result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, (
        "host conformance executable failed:\n"
        f"{run_result.stdout}{run_result.stderr}"
    )

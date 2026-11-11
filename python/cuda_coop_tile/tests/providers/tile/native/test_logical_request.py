# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT


def test_logical_request_binds_and_owns_validated_abi_values(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("host ABI test requires a C++17 compiler")
    configured = os.environ.get("CUDA_COOP_TILE_SDK_ROOT")
    if not configured:
        pytest.skip("host ABI test requires CUDA_COOP_TILE_SDK_ROOT")
    include = Path(configured) / "include"
    assert (include / "tile_ir/linkage_provider/LinkageProviderAPI.h").is_file()
    native = PACKAGE_ROOT.parents[1] / "tile/cpp/tilelib"
    sources = [
        native / "provider" / name
        for name in (
            "metadata_parser.cpp",
            "primitive_spec.cpp",
            "provider_request.cpp",
            "registry.cpp",
            "source_common.cpp",
            "spec_identity.cpp",
            "runtime/cache_keys.cpp",
        )
    ]
    sources.extend(sorted((native / "provider/families").glob("*.cpp")))
    executable = tmp_path / "logical_request"
    command = [
        compiler,
        "-std=c++17",
        "-O0",
        "-ffunction-sections",
        "-fdata-sections",
        f"-I{native}",
        f"-I{include}",
        str(
            PACKAGE_ROOT.parents[1]
            / "tile/cpp/test/support/logical_request.cpp"
        ),
        *map(str, sources),
        "-Wl,--gc-sections",
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(
        command, check=False, text=True, capture_output=True
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    checked = subprocess.run(
        [str(executable)], check=False, text=True, capture_output=True
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

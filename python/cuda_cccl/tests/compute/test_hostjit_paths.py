# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute as compute
import cuda.compute._build_info as build_info
import cuda.compute._cccl_interop as cccl_interop

pytestmark = pytest.mark.no_numba

_HOSTJIT_ENVIRONMENT = ("HOSTJIT_CLANG_PATH", "HOSTJIT_INCLUDE_PATH")


def _clear_hostjit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Seed then delete each variable so monkeypatch records the absent state and
    # can restore direct writes made by production code after the test.
    for name in _HOSTJIT_ENVIRONMENT:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _create_hostjit_bundle(package_dir: Path) -> Path:
    hostjit_dir = package_dir / "_cccl"
    clang_header = hostjit_dir / "clang" / "__clang_cuda_math_forward_declares.h"
    runtime_header = (
        hostjit_dir / "hostjit" / "cuda_minimal" / "__clang_cuda_runtime_wrapper.h"
    )
    clang_header.parent.mkdir(parents=True)
    runtime_header.parent.mkdir(parents=True)
    clang_header.touch()
    runtime_header.touch()
    return hostjit_dir


def test_configure_hostjit_paths_uses_compute_private_bundle(
    monkeypatch, tmp_path
) -> None:
    package_dir = tmp_path / "cuda" / "compute"
    hostjit_dir = _create_hostjit_bundle(package_dir)
    monkeypatch.setattr(compute, "__file__", str(package_dir / "__init__.py"))
    monkeypatch.setattr(build_info, "USING_V2", True)
    _clear_hostjit_environment(monkeypatch)

    compute._configure_hostjit_paths()

    assert os.environ["HOSTJIT_CLANG_PATH"] == str(hostjit_dir / "clang")
    assert os.environ["HOSTJIT_INCLUDE_PATH"] == str(hostjit_dir)


def test_configure_hostjit_paths_preserves_explicit_configuration(
    monkeypatch, tmp_path
) -> None:
    package_dir = tmp_path / "cuda" / "compute"
    _create_hostjit_bundle(package_dir)
    monkeypatch.setattr(compute, "__file__", str(package_dir / "__init__.py"))
    monkeypatch.setattr(build_info, "USING_V2", True)
    monkeypatch.setenv("HOSTJIT_CLANG_PATH", "/custom/clang")
    monkeypatch.setenv("HOSTJIT_INCLUDE_PATH", "/custom/hostjit")

    compute._configure_hostjit_paths()

    assert os.environ["HOSTJIT_CLANG_PATH"] == "/custom/clang"
    assert os.environ["HOSTJIT_INCLUDE_PATH"] == "/custom/hostjit"


def test_configure_hostjit_paths_ignores_incomplete_private_bundle(
    monkeypatch, tmp_path
) -> None:
    package_dir = tmp_path / "cuda" / "compute"
    (package_dir / "_cccl" / "clang").mkdir(parents=True)
    monkeypatch.setattr(compute, "__file__", str(package_dir / "__init__.py"))
    monkeypatch.setattr(build_info, "USING_V2", True)
    _clear_hostjit_environment(monkeypatch)

    compute._configure_hostjit_paths()

    assert "HOSTJIT_CLANG_PATH" not in os.environ
    assert "HOSTJIT_INCLUDE_PATH" not in os.environ


@pytest.mark.skipif(
    not build_info.USING_V2 or build_info.BUILD_STATE != "wheel",
    reason="requires an installed cuda-compute wheel using the v2 HostJIT backend",
)
@pytest.mark.parametrize("include_suffix", ["", os.sep])
def test_v2_c_api_derives_runtime_paths_from_private_headers(
    monkeypatch, include_suffix: str
) -> None:
    _clear_hostjit_environment(monkeypatch)
    paths = cccl_interop.get_include_paths()
    test_paths = replace(
        paths,
        libcudacxx=f"{paths.libcudacxx}{include_suffix}",
    )
    monkeypatch.setattr(cccl_interop, "get_include_paths", lambda: test_paths)
    compute.clear_all_caches()

    h_input = np.arange(1, 8, dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, np.int32)
    h_init = np.array([3], dtype=np.int32)

    compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=h_input.size,
        op=compute.OpKind.PLUS,
        h_init=h_init,
    )

    assert d_output.copy_to_host()[0] == np.sum(h_input, initial=h_init[0])

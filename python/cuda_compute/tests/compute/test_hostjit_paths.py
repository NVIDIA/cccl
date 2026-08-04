# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path

import cuda.compute as compute
import cuda.compute._build_info as build_info


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
    monkeypatch.delenv("HOSTJIT_CLANG_PATH", raising=False)
    monkeypatch.delenv("HOSTJIT_INCLUDE_PATH", raising=False)

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

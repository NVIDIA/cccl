# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CMake configuration contracts for full and narrow source trees."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ...support.paths import PACKAGE_ROOT


def _require_cmake() -> str:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to validate cuda-coop configuration")
    return cmake


def _narrow_source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "narrow-cccl" / "python" / "cuda_coop"
    source.mkdir(parents=True)
    shutil.copy2(PACKAGE_ROOT / "CMakeLists.txt", source / "CMakeLists.txt")
    assert not (source.parents[1] / "cmake" / "CCCLInstallRules.cmake").exists()
    return source


def _configure_narrow_source(
    cmake: str,
    source: Path,
    build: Path,
    *options: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cmake, "-S", str(source), "-B", str(build), *options],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_narrow_source_configures_without_cccl_header_bundle(tmp_path):
    cmake = _require_cmake()
    source = _narrow_source_tree(tmp_path)
    build = tmp_path / "build"
    install = tmp_path / "install"

    configured = _configure_narrow_source(
        cmake,
        source,
        build,
        "-DCUDA_COOP_INSTALL_HEADER_BUNDLE=OFF",
    )

    assert configured.returncode == 0, configured.stdout + configured.stderr
    installed = subprocess.run(
        [cmake, "--install", str(build), "--prefix", str(install)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    assert not (install / "cuda" / "coop" / "_headers").exists()


def test_narrow_source_rejects_header_bundle_with_actionable_error(tmp_path):
    cmake = _require_cmake()
    source = _narrow_source_tree(tmp_path)

    configured = _configure_narrow_source(cmake, source, tmp_path / "build")

    diagnostic = configured.stdout + configured.stderr
    assert configured.returncode != 0
    assert (
        "CUDA_COOP_INSTALL_HEADER_BUNDLE=ON requires a full CCCL source tree"
        in diagnostic
    )
    assert "-DCUDA_COOP_INSTALL_HEADER_BUNDLE=OFF" in diagnostic


def test_full_source_configure_records_a_revision_token(tmp_path):
    cmake = _require_cmake()
    build = tmp_path / "build"

    configured = subprocess.run(
        [
            cmake,
            "-S",
            str(PACKAGE_ROOT),
            "-B",
            str(build),
            "-DCUDA_COOP_CCCL_SOURCE_REVISION=not a token",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    diagnostic = configured.stdout + configured.stderr
    assert configured.returncode != 0
    assert "must be a revision token" in diagnostic

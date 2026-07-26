# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for merge_cuda_wheels.py."""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

# merge_cuda_wheels.py is a standalone script, not an installed package.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "merge_cuda_wheels", Path(__file__).parents[1] / "merge_cuda_wheels.py"
)
merge_cuda_wheels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_cuda_wheels)
merge_wheels = merge_cuda_wheels.merge_wheels


def _wheel_pack(src_dir: Path, dest_dir: Path) -> Path:
    """Run `wheel pack` and return the produced wheel path."""
    subprocess.run(
        [sys.executable, "-m", "wheel", "pack", str(src_dir), "--dest-dir", str(dest_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(dest_dir.glob(f"{src_dir.name}*.whl"))
    assert len(wheels) == 1, f"Expected one wheel for {src_dir.name}, got {wheels}"
    return wheels[0]


def _make_synthetic_wheel(dest_dir: Path, dist_name: str, version: str, cuda_version: int) -> Path:
    """Build a minimal CUDA-specific wheel with the requested compute directory."""
    work_dir = dest_dir / f"build_cu{cuda_version}"
    work_dir.mkdir(parents=True)
    pkg_dir = work_dir / f"{dist_name}-{version}"
    compute_dir = pkg_dir / "cuda" / "compute" / f"cu{cuda_version}"
    compute_dir.mkdir(parents=True)
    (compute_dir / "lib.so").write_text(f"payload cu{cuda_version}")
    (pkg_dir / "cuda" / "__init__.py").write_text("")
    (pkg_dir / "cuda" / "compute" / "__init__.py").write_text("")

    dist_info = pkg_dir / f"{dist_name}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    (dist_info / "METADATA").write_text(f"Name: {dist_name.replace('_', '-')}\nVersion: {version}\n")

    return _wheel_pack(pkg_dir, work_dir)


def _make_decoy_wheel(dest_dir: Path, name: str) -> Path:
    """Build an unrelated wheel to seed the output directory."""
    pkg_dir = dest_dir / f"{name}-1.0.0"
    (pkg_dir / name).mkdir(parents=True)
    (pkg_dir / name / "__init__.py").write_text("")
    dist_info = pkg_dir / f"{name}-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    (dist_info / "METADATA").write_text(f"Name: {name.replace('_', '-')}\nVersion: 1.0.0\n")
    return _wheel_pack(pkg_dir, dest_dir)


def test_merge_wheels_returns_expected_wheel_despite_decoys():
    """merge_wheels must return the deterministic merged wheel, not a glob hit.

    Before the fix, `output_dir.glob('*.whl')[0]` could return a pre-existing
    decoy wheel when the output directory was not empty.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        output_dir = tmp_path / "merged"
        output_dir.mkdir()

        cu12_wheel = _make_synthetic_wheel(build_dir, "cuda_cccl", "1.0.0", 12)
        cu13_wheel = _make_synthetic_wheel(build_dir, "cuda_cccl", "1.0.0", 13)

        # The merger expects CUDA-specific suffixes on the input wheel names.
        cu12_wheel = cu12_wheel.rename(
            cu12_wheel.with_name(cu12_wheel.name.replace(".whl", ".cu12.whl"))
        )
        cu13_wheel = cu13_wheel.rename(
            cu13_wheel.with_name(cu13_wheel.name.replace(".whl", ".cu13.whl"))
        )

        # Seed the output directory with decoy wheels created before the merge.
        for i in range(5):
            _make_decoy_wheel(output_dir, f"aa_decoy_{i}")

        merged_wheel = merge_wheels([cu12_wheel, cu13_wheel], output_dir)

        expected = output_dir / "cuda_cccl-1.0.0-py3-none-any.whl"
        assert merged_wheel == expected, f"Expected {expected}, got {merged_wheel}"
        assert merged_wheel.exists()

        with zipfile.ZipFile(merged_wheel) as zf:
            names = zf.namelist()
            assert "cuda/compute/cu12/lib.so" in names
            assert "cuda/compute/cu13/lib.so" in names

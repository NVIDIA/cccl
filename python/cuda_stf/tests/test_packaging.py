# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Packaging invariants for installing cuda_stf alongside cuda_cccl."""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

_PYTHON_DIR = Path(__file__).resolve().parents[2]
_PYPROJECTS = {
    "cuda_cccl": _PYTHON_DIR / "cuda_cccl" / "pyproject.toml",
    "cuda_stf": _PYTHON_DIR / "cuda_stf" / "pyproject.toml",
}


def _minimal_cu12_requirements(package: str) -> list[str]:
    path = _PYPROJECTS[package]
    if not path.is_file():
        pytest.skip(f"{path} is not available (not running from a source checkout)")
    data = tomllib.loads(path.read_text())
    return data["project"]["optional-dependencies"]["minimal-cu12"]


@pytest.mark.parametrize("package", sorted(_PYPROJECTS))
def test_nvjitlink_is_requested_directly(package: str):
    """nvjitlink must not be taken through the ``cuda-toolkit`` extra.

    That extra hard-pins ``nvidia-nvjitlink-cu12`` to the toolkit's exact
    version. cuda_cccl requires numba-cuda-mlir, which needs
    ``nvidia-nvjitlink-cu12>=12.3.0``, and both packages are installed in a
    single resolve, so a hard pin in either one makes that install
    unsatisfiable against an older pinned toolkit.
    """
    requirements = _minimal_cu12_requirements(package)

    toolkit_requirements = [r for r in requirements if r.startswith("cuda-toolkit[")]
    assert toolkit_requirements, f"{package} has no cuda-toolkit requirement"
    for requirement in toolkit_requirements:
        extras = requirement[requirement.index("[") + 1 : requirement.index("]")]
        assert "nvjitlink" not in [e.strip() for e in extras.split(",")], (
            f"{package} takes nvjitlink via {requirement!r}, which hard-pins it"
        )

    assert any(r.startswith("nvidia-nvjitlink-cu12") for r in requirements), (
        f"{package} must request nvidia-nvjitlink-cu12 directly"
    )

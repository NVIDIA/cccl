# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Packaging invariants for cuda_cccl's dependency declarations."""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _optional_dependencies() -> dict[str, list[str]]:
    if not _PYPROJECT.is_file():
        pytest.skip(f"{_PYPROJECT} is not available (not a source checkout)")
    # Read as binary: TOML is always UTF-8, whereas decoding text applies the
    # locale's encoding, which fails on a non-ASCII character under a C locale.
    with _PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def test_jit_backend_pin_is_bounded():
    """numba-cuda-mlir must carry an upper bound.

    Reaching the LLVM bitcode the v2 backend needs uses numba-cuda-mlir
    internals, so a minor release has to be validated before it is allowed.
    """
    extras = _optional_dependencies()

    pins = [
        requirement
        for requirements in extras.values()
        for requirement in requirements
        if requirement.startswith("numba-cuda-mlir")
    ]
    assert pins, "no extra declares numba-cuda-mlir"

    for pin in pins:
        assert "<" in pin, f"{pin!r} has no upper bound"

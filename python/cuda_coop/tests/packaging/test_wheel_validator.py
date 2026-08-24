# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_validator():
    validator_path = Path(__file__).parents[4] / "ci" / "validate_cuda_coop_wheel.py"
    spec = importlib.util.spec_from_file_location(
        "validate_cuda_coop_wheel",
        validator_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_obsolete_layout_check_ignores_bundled_headers() -> None:
    validator = _load_validator()

    validator._reject_obsolete_layouts(
        {"cuda/coop/_headers/include/cub/_block/detail.cuh"}
    )


def test_obsolete_layout_check_rejects_nested_python_packages() -> None:
    validator = _load_validator()

    with pytest.raises(SystemExit, match="cuda/coop/cutlass/_dsl/__init__.py"):
        validator._reject_obsolete_layouts({"cuda/coop/cutlass/_dsl/__init__.py"})

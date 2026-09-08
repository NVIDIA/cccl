# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Validate Store scalars with compiler types, including unresolved provenance."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as root_coop

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel, *arg_types):
    # Configured launches are the runtime's only route to exact block metadata.
    result = kernel._compile_launch_config_signature(
        types.void(*arg_types),
        (
            ("grid", (1, 1, 1)),
            ("block", (32, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )
    assert result.metadata["ltoir"]
    return result


@pytest.mark.parametrize("coop", (root_coop, qualified_coop), ids=("root", "qualified"))
@pytest.mark.parametrize("expression", ("abs", "min", "loop"))
@pytest.mark.parametrize("matching", (False, True), ids=("mismatch", "exact"))
def test_store_checks_actual_expression_dtype(coop, expression, matching):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        value = source[cuda.threadIdx.x]
        if expression == "abs":
            result = abs(value)
        elif expression == "min":
            result = min(value, 3.5)
        else:
            result = value
            for _ in range(2):
                result += np.float32(1.0)
        coop.store(coop.this_block(), destination, result)

    result_dtype = types.float64 if expression == "min" else types.float32
    destination_dtype = result_dtype if matching else types.int32
    arg_types = (types.float32[::1], destination_dtype[::1])
    if matching:
        _compile(kernel, *arg_types)
    else:
        with pytest.raises(TypingError, match="does not match destination dtype"):
            _compile(kernel, *arg_types)


@pytest.mark.parametrize("coop", (root_coop, qualified_coop), ids=("root", "qualified"))
@pytest.mark.parametrize(
    ("value", "dtype"),
    (
        (127, types.int8),
        (-128, types.int8),
        (3.5, types.float32),
        (255, types.uint8),
    ),
)
def test_store_preserves_contextual_literal_conversion(coop, value, dtype):
    @cuda.jit(chip="sm_90")
    def kernel(destination):
        coop.store(coop.this_block(), destination, value)

    _compile(kernel, dtype[::1])

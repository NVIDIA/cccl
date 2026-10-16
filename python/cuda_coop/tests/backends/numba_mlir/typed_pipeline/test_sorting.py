# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile MergeSort ownership, callbacks and sentinel precision."""

from types import SimpleNamespace

import numba_cuda_mlir.tools as compiler_tools
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

from ._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline


@pytest.fixture(autouse=True)
def _hidden_target(monkeypatch):
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel, signature):
    result = cuda.jit(chip="sm_90")(kernel)._compile_launch_config_signature(
        signature,
        (
            ("grid", (1, 1, 1)),
            ("block", (64, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )
    assert result.metadata["cubin"]
    return result


def test_merge_comparator_links_with_infinite_partial_sentinel():
    sentinel = float("-inf")

    def greater(a, b):
        return a > b

    def kernel(source, output, count):
        group = coop.this_block()
        keys = coop.ThreadData(2)
        coop.load(group, source, keys)
        result = numba_coop.merge_sort_keys(
            group,
            keys,
            compare_op=greater,
            valid_items=count,
            oob_default=sentinel,
        )
        coop.store(group, output, result)

    _compile(
        kernel, types.void(types.float64[::1], types.float64[::1], types.int64)
    )


@pytest.mark.parametrize("dtype", [types.int32, types.int64])
def test_merge_literal_argument_retains_its_sentinel_dtype(dtype):
    def kernel(source, output, sentinel):
        group = coop.this_block()
        keys = coop.ThreadData(2)
        coop.load(group, source, keys)
        result = coop.merge_sort_keys(
            group, keys, valid_items=64, oob_default=sentinel
        )
        coop.store(group, output, result)

    signature = types.void(dtype[::1], dtype[::1], types.IntegerLiteral(1000))
    if dtype == types.int32:
        with pytest.raises(TypingError, match="oob_default dtype int64"):
            _compile(kernel, signature)
    else:
        _compile(kernel, signature)

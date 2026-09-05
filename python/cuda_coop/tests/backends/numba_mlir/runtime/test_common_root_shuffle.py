# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS_PER_THREAD = 3
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_SEGMENTS = 4
_SENTINEL = np.int32(-999_999)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_shuffle_kernel(source, output):
    tid = (
        cuda.threadIdx.x
        + cuda.threadIdx.y * cuda.blockDim.x
        + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    )
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]

    group = coop.this_block()
    default_down = coop.shuffle(group, items)
    explicit_down = coop.shuffle(group, items, mode="down", distance=1)
    explicit_up = coop.shuffle(group, items, mode="up", distance=1)

    for index in range(_ITEMS_PER_THREAD):
        item = tid * _ITEMS_PER_THREAD + index
        output[0 * _TILE_ITEMS + item] = items[index]
        if item + 1 < _TILE_ITEMS:
            output[1 * _TILE_ITEMS + item] = default_down[index]
            output[2 * _TILE_ITEMS + item] = explicit_down[index]
        if item > 0:
            output[3 * _TILE_ITEMS + item] = explicit_up[index]


@cuda.jit
def _qualified_shuffle_kernel(source, output):
    tid = (
        cuda.threadIdx.x
        + cuda.threadIdx.y * cuda.blockDim.x
        + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    )
    items = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]

    group = numba_coop.this_block()
    default_down = numba_coop.shuffle(group, items)
    explicit_down = numba_coop.shuffle(group, items, mode="down", distance=1)
    explicit_up = numba_coop.shuffle(group, items, mode="up", distance=1)

    for index in range(_ITEMS_PER_THREAD):
        item = tid * _ITEMS_PER_THREAD + index
        output[0 * _TILE_ITEMS + item] = items[index]
        if item + 1 < _TILE_ITEMS:
            output[1 * _TILE_ITEMS + item] = default_down[index]
            output[2 * _TILE_ITEMS + item] = explicit_down[index]
        if item > 0:
            output[3 * _TILE_ITEMS + item] = explicit_up[index]


def _expected_shuffle(values):
    expected = np.full((_SEGMENTS, _TILE_ITEMS), _SENTINEL, dtype=np.int32)
    expected[0] = values
    expected[1, :-1] = values[1:]
    expected[2, :-1] = values[1:]
    expected[3, 1:] = values[:-1]
    return expected.reshape(-1)


def _run_and_check(values):
    original = values.copy()
    common_output = np.full(_SEGMENTS * _TILE_ITEMS, _SENTINEL, dtype=np.int32)
    qualified_output = np.full_like(common_output, _SENTINEL)

    _common_shuffle_kernel[1, _BLOCK](values, common_output)
    _qualified_shuffle_kernel[1, _BLOCK](values, qualified_output)
    cuda.synchronize()

    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(common_output, qualified_output)
    np.testing.assert_array_equal(common_output, _expected_shuffle(original))


@pytest.mark.evidence_for("group.shuffle", backend="numba_mlir", evidence="runtime")
def test_common_shuffle_matches_qualified_numba_and_independent_oracle_twice():
    indices = np.arange(_TILE_ITEMS, dtype=np.int32)
    _run_and_check(((indices * 17) % 109) - 54)
    _run_and_check(((indices * -13) % 97) + 11)


@cuda.jit
def _qualified_complex_shuffle_kernel(source, down_output, up_output):
    tid = cuda.threadIdx.x
    items = numba_coop.ThreadData(2, dtype=types.complex128)
    for index in range(2):
        items[index] = source[tid * 2 + index]

    group = numba_coop.this_block()
    down = numba_coop.shuffle(group, items, mode="down")
    up = numba_coop.shuffle(group, items, mode="up")

    for index in range(2):
        item = tid * 2 + index
        if item + 1 < _THREADS * 2:
            down_output[item] = down[index]
        if item > 0:
            up_output[item] = up[index]


def test_qualified_shuffle_supports_complex128_aggregate_payloads():
    item_count = _THREADS * 2
    real = np.arange(item_count, dtype=np.float64)
    values = (real + 1j * (real * 3.0 + 0.5)).astype(np.complex128)
    sentinel = np.complex128(-991.0 - 773.0j)
    down = np.full(item_count, sentinel, dtype=np.complex128)
    up = np.full(item_count, sentinel, dtype=np.complex128)

    _qualified_complex_shuffle_kernel[1, _THREADS](values, down, up)
    cuda.synchronize()

    expected_down = np.full_like(values, sentinel)
    expected_down[:-1] = values[1:]
    expected_up = np.full_like(values, sentinel)
    expected_up[1:] = values[:-1]
    np.testing.assert_array_equal(down, expected_down)
    np.testing.assert_array_equal(up, expected_up)

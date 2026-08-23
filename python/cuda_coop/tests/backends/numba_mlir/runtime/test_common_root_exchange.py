# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_THREADS = 64
_WARP_THREADS = 32
_ITEMS_PER_THREAD = 5
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_SEGMENTS = 5
_COMPLEX_ITEMS_PER_THREAD = 2
_COMPLEX_TILE_ITEMS = _THREADS * _COMPLEX_ITEMS_PER_THREAD

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_exchange_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = d_input[tid * _ITEMS_PER_THREAD + index]

    block = coop.this_block()
    block_blocked = coop.exchange(block, items)
    block_striped = coop.exchange(block, items, mode="blocked_to_striped")
    warp = coop.this_warp()
    warp_blocked = coop.exchange(warp, items)
    warp_striped = coop.exchange(warp, items, mode="blocked_to_striped")

    for index in range(_ITEMS_PER_THREAD):
        item = tid * _ITEMS_PER_THREAD + index
        d_output[0 * _TILE_ITEMS + item] = items[index]
        d_output[1 * _TILE_ITEMS + item] = block_blocked[index]
        d_output[2 * _TILE_ITEMS + item] = block_striped[index]
        d_output[3 * _TILE_ITEMS + item] = warp_blocked[index]
        d_output[4 * _TILE_ITEMS + item] = warp_striped[index]


@cuda.jit
def _qualified_exchange_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    items = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = d_input[tid * _ITEMS_PER_THREAD + index]

    block = numba_coop.this_block()
    block_blocked = numba_coop.exchange(block, items)
    block_striped = numba_coop.exchange(block, items, mode="blocked_to_striped")
    warp = numba_coop.this_warp()
    warp_blocked = numba_coop.exchange(warp, items)
    warp_striped = numba_coop.exchange(warp, items, mode="blocked_to_striped")

    for index in range(_ITEMS_PER_THREAD):
        item = tid * _ITEMS_PER_THREAD + index
        d_output[0 * _TILE_ITEMS + item] = items[index]
        d_output[1 * _TILE_ITEMS + item] = block_blocked[index]
        d_output[2 * _TILE_ITEMS + item] = block_striped[index]
        d_output[3 * _TILE_ITEMS + item] = warp_blocked[index]
        d_output[4 * _TILE_ITEMS + item] = warp_striped[index]


@cuda.jit
def _qualified_complex_exchange_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    items = numba_coop.ThreadData(
        _COMPLEX_ITEMS_PER_THREAD,
        dtype=types.complex128,
    )
    for index in range(_COMPLEX_ITEMS_PER_THREAD):
        items[index] = d_input[tid * _COMPLEX_ITEMS_PER_THREAD + index]

    blocked = numba_coop.exchange(numba_coop.this_block(), items)
    warp_blocked = numba_coop.exchange(numba_coop.this_warp(), items)
    for index in range(_COMPLEX_ITEMS_PER_THREAD):
        item = tid * _COMPLEX_ITEMS_PER_THREAD + index
        d_output[0 * _COMPLEX_TILE_ITEMS + item] = items[index]
        d_output[1 * _COMPLEX_TILE_ITEMS + item] = blocked[index]
        d_output[2 * _COMPLEX_TILE_ITEMS + item] = warp_blocked[index]


def _expected_exchange(values):
    expected = np.empty((_SEGMENTS, _TILE_ITEMS), dtype=np.int32)
    expected[0] = values
    expected[1] = values.reshape(_THREADS, _ITEMS_PER_THREAD).T.reshape(-1)
    expected[2] = values.reshape(_ITEMS_PER_THREAD, _THREADS).T.reshape(-1)

    warp_items = _WARP_THREADS * _ITEMS_PER_THREAD
    expected[3] = np.concatenate(
        [
            values[begin : begin + warp_items]
            .reshape(_WARP_THREADS, _ITEMS_PER_THREAD)
            .T.reshape(-1)
            for begin in range(0, _TILE_ITEMS, warp_items)
        ]
    )
    expected[4] = np.concatenate(
        [
            values[begin : begin + warp_items]
            .reshape(_ITEMS_PER_THREAD, _WARP_THREADS)
            .T.reshape(-1)
            for begin in range(0, _TILE_ITEMS, warp_items)
        ]
    )
    return expected.reshape(-1)


@pytest.mark.evidence_for("group.exchange", backend="numba_mlir", evidence="runtime")
def test_common_exchange_matches_qualified_numba_and_independent_oracle():
    values = ((np.arange(_TILE_ITEMS, dtype=np.int32) * 17) % 103) - 51
    common_output = np.full(_SEGMENTS * _TILE_ITEMS, -999, dtype=np.int32)
    qualified_output = np.full_like(common_output, -999)

    _common_exchange_kernel[1, _THREADS](values, common_output)
    _qualified_exchange_kernel[1, _THREADS](values, qualified_output)
    cuda.synchronize()

    np.testing.assert_array_equal(common_output, qualified_output)
    np.testing.assert_array_equal(common_output, _expected_exchange(values))


def test_qualified_complex_exchange_matches_independent_oracle():
    indices = np.arange(_COMPLEX_TILE_ITEMS, dtype=np.float64)
    values = (indices * 1.25 + 0.5) + (indices * -0.75 + 3.0) * 1j
    qualified_output = np.zeros(3 * _COMPLEX_TILE_ITEMS, dtype=np.complex128)

    _qualified_complex_exchange_kernel[1, _THREADS](values, qualified_output)
    cuda.synchronize()

    expected = np.empty((3, _COMPLEX_TILE_ITEMS), dtype=np.complex128)
    expected[0] = values
    expected[1] = values.reshape(_THREADS, _COMPLEX_ITEMS_PER_THREAD).T.reshape(-1)
    warp_items = _WARP_THREADS * _COMPLEX_ITEMS_PER_THREAD
    expected[2] = np.concatenate(
        [
            values[begin : begin + warp_items]
            .reshape(_WARP_THREADS, _COMPLEX_ITEMS_PER_THREAD)
            .T.reshape(-1)
            for begin in range(0, _COMPLEX_TILE_ITEMS, warp_items)
        ]
    )

    np.testing.assert_array_equal(qualified_output, expected.reshape(-1))

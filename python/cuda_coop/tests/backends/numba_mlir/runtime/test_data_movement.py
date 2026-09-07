# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as _numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 64
_ITEMS_PER_THREAD = 2
_WIDE_ITEMS_PER_THREAD = 8
_LOGICAL_WARP_THREADS = 8


@cuda.jit
def _block_load_store_fixed_storage(source, output):
    storage = coop.TempStorage(4096, alignment=16)
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded = coop.load(
        coop.this_block(),
        source,
        items,
        algorithm="transpose",
        temp_storage=storage,
    )
    coop.store(
        coop.this_block(),
        output,
        loaded,
        algorithm="transpose",
        temp_storage=storage,
    )


def test_block_load_store_forwards_fixed_capacity_temp_storage():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _block_load_store_fixed_storage[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, source)


@cuda.jit
def _logical_warp_load_store(source, output):
    logical_warp = coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded = coop.load(logical_warp, source, items)
    coop.store(logical_warp, output, loaded)


def test_logical_width_8_load_store_round_trip():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _logical_warp_load_store[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, source)


@cuda.jit
def _logical_warp_exchange_round_trip(source, output):
    tid = cuda.threadIdx.x
    logical_warp = coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]
    striped = coop.exchange(logical_warp, items, mode="blocked_to_striped")
    blocked = coop.exchange(logical_warp, striped, mode="striped_to_blocked")
    for index in range(_ITEMS_PER_THREAD):
        output[tid * _ITEMS_PER_THREAD + index] = blocked[index]


def test_exchange_round_trip_with_logical_width_8():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _logical_warp_exchange_round_trip[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, source)


@cuda.jit
def _block_exchange_eight_items(source, output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_WIDE_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_WIDE_ITEMS_PER_THREAD):
        items[index] = source[tid * _WIDE_ITEMS_PER_THREAD + index]
    striped = coop.exchange(coop.this_block(), items, mode="blocked_to_striped")
    blocked = coop.exchange(coop.this_block(), striped, mode="striped_to_blocked")
    for index in range(_WIDE_ITEMS_PER_THREAD):
        output[tid * _WIDE_ITEMS_PER_THREAD + index] = blocked[index]


def test_common_exchange_accepts_eight_items_per_thread():
    source = np.arange(_THREADS * _WIDE_ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _block_exchange_eight_items[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, source)


@cuda.jit
def _time_sliced_block_exchange(source, output):
    tid = cuda.threadIdx.x
    items = _numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]
    striped = _numba_coop.exchange(
        _numba_coop.this_block(),
        items,
        mode="blocked_to_striped",
        warp_time_slicing=True,
    )
    blocked = _numba_coop.exchange(
        _numba_coop.this_block(),
        striped,
        mode="striped_to_blocked",
        warp_time_slicing=True,
    )
    for index in range(_ITEMS_PER_THREAD):
        output[tid * _ITEMS_PER_THREAD + index] = blocked[index]


def test_time_sliced_exchange_round_trip():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _time_sliced_block_exchange[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, source)


@cuda.jit
def _block_shuffle_down(source, output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]
    shifted = coop.shuffle(coop.this_block(), items)
    for index in range(_ITEMS_PER_THREAD):
        position = tid * _ITEMS_PER_THREAD + index
        if position + 1 < _THREADS * _ITEMS_PER_THREAD:
            output[position] = shifted[index]


def test_block_shuffle_down_moves_one_item():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _block_shuffle_down[1, _THREADS](source, output)

    np.testing.assert_array_equal(output[:-1], source[1:])
    assert output[-1] == -1

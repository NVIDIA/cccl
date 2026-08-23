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

import cuda.coop.numba_mlir as coop
from cuda import coop as common_coop

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
_TOTAL_ITEMS = _THREADS * _ITEMS_PER_THREAD
_BLOCK_VALID_ITEMS = _TOTAL_ITEMS - 11
_LOGICAL_WARP_THREADS = 8
_LOGICAL_WARP_ITEMS = _LOGICAL_WARP_THREADS * _ITEMS_PER_THREAD
_LOGICAL_WARP_VALID_ITEMS = _LOGICAL_WARP_ITEMS - 3
_INT32_MAX = np.int32(2_147_483_647)


@cuda.jit(device=True)
def _less(lhs, rhs):
    return lhs < rhs


@cuda.jit
def _block_merge_sort(
    source_keys,
    source_values,
    original_keys,
    sorted_pair_keys,
    sorted_pair_values,
    descending_keys,
    valid_items,
    oob_default,
):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(_ITEMS_PER_THREAD)
    values = coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = source_keys[index]
        values[item] = source_values[index]

    fixed_storage = coop.TempStorage(4096, alignment=16)
    pair_result = coop.merge_sort_pairs(
        coop.this_block(),
        keys,
        values,
        compare_op=_less,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=fixed_storage,
    )
    deferred_storage = coop.TempStorage()
    descending_result = coop.merge_sort_keys(
        coop.this_block(),
        keys,
        descending=True,
        temp_storage=deferred_storage,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        original_keys[index] = keys[item]
        sorted_pair_keys[index] = pair_result[0][item]
        sorted_pair_values[index] = pair_result[1][item]
        descending_keys[index] = descending_result[item]


def test_block_keys_and_pairs_use_fresh_results_and_both_storage_forms():
    indices = np.arange(_TOTAL_ITEMS, dtype=np.int32)
    keys = ((indices * np.int32(53)) % np.int32(_TOTAL_ITEMS)) - np.int32(64)
    values = indices.astype(np.float32) + np.float32(0.25)
    original = np.full_like(keys, -9999)
    pair_keys = np.full_like(keys, -9999)
    pair_values = np.full_like(values, -9999.0)
    descending = np.full_like(keys, -9999)

    _block_merge_sort[1, _THREADS](
        keys,
        values,
        original,
        pair_keys,
        pair_values,
        descending,
        np.int32(_BLOCK_VALID_ITEMS),
        np.int32(np.iinfo(np.int32).max),
    )

    np.testing.assert_array_equal(original, keys)
    order = np.argsort(keys[:_BLOCK_VALID_ITEMS])
    np.testing.assert_array_equal(
        pair_keys[:_BLOCK_VALID_ITEMS],
        keys[:_BLOCK_VALID_ITEMS][order],
    )
    np.testing.assert_array_equal(
        pair_values[:_BLOCK_VALID_ITEMS],
        values[:_BLOCK_VALID_ITEMS][order],
    )
    np.testing.assert_array_equal(descending, np.sort(keys)[::-1])


@cuda.jit
def _common_block_merge_sort(source, output):
    tid = cuda.threadIdx.x
    keys = common_coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = source[index]
    result = common_coop.merge_sort_keys(common_coop.this_block(), keys)
    for item in range(_ITEMS_PER_THREAD):
        output[tid * _ITEMS_PER_THREAD + item] = result[item]


def test_common_root_block_merge_sort_uses_the_numba_backend():
    indices = np.arange(_TOTAL_ITEMS, dtype=np.int32)
    keys = (indices * np.int32(53)) % np.int32(_TOTAL_ITEMS)
    output = np.full_like(keys, -1)

    _common_block_merge_sort[1, _THREADS](keys, output)

    np.testing.assert_array_equal(output, np.sort(keys))


@cuda.jit
def _physical_warp_scalar_merge_sort(source, output):
    tid = cuda.threadIdx.x
    output[tid] = coop.merge_sort_keys(
        coop.this_warp(),
        source[tid],
        descending=True,
    )


def test_physical_warp_scalar_keys_sort_each_warp():
    indices = np.arange(_THREADS, dtype=np.int32)
    keys = ((indices * np.int32(29)) % np.int32(_THREADS)) - np.int32(32)
    output = np.full_like(keys, -9999)

    _physical_warp_scalar_merge_sort[1, _THREADS](keys, output)

    for base in range(0, _THREADS, 32):
        end = base + 32
        np.testing.assert_array_equal(output[base:end], np.sort(keys[base:end])[::-1])


@cuda.jit
def _physical_warp_thread_data_merge_sort(source, output):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = source[index]
    result = coop.merge_sort_keys(coop.this_warp(), keys)
    for item in range(_ITEMS_PER_THREAD):
        output[tid * _ITEMS_PER_THREAD + item] = result[item]


def test_physical_warp_thread_data_keys_infer_dtype_from_writes():
    indices = np.arange(_TOTAL_ITEMS, dtype=np.int32)
    keys = ((indices * np.int32(29)) % np.int32(64)) - np.int32(32)
    output = np.full_like(keys, -9999)

    _physical_warp_thread_data_merge_sort[1, _THREADS](keys, output)

    for base in range(0, _TOTAL_ITEMS, 32 * _ITEMS_PER_THREAD):
        end = base + 32 * _ITEMS_PER_THREAD
        np.testing.assert_array_equal(output[base:end], np.sort(keys[base:end]))


@cuda.jit
def _logical_warp_pair_merge_sort(
    source_keys,
    source_values,
    original_keys,
    sorted_keys,
    sorted_values,
):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(_ITEMS_PER_THREAD)
    values = coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = source_keys[index]
        values[item] = source_values[index]

    logical_warp = coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    result = coop.merge_sort_pairs(
        logical_warp,
        keys,
        values,
        valid_items=_LOGICAL_WARP_VALID_ITEMS,
        oob_default=_INT32_MAX,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        original_keys[index] = keys[item]
        sorted_keys[index] = result[0][item]
        sorted_values[index] = result[1][item]


def test_logical_warp_thread_data_pairs_preserve_association():
    indices = np.arange(_TOTAL_ITEMS, dtype=np.int32)
    keys = np.empty_like(indices)
    for base in range(0, _TOTAL_ITEMS, _LOGICAL_WARP_ITEMS):
        local = np.arange(_LOGICAL_WARP_ITEMS, dtype=np.int32)
        keys[base : base + _LOGICAL_WARP_ITEMS] = (
            (local * np.int32(7)) % np.int32(_LOGICAL_WARP_ITEMS)
        ) - np.int32(8)
    values = indices * np.int32(17) + np.int32(3)
    original = np.full_like(keys, -9999)
    sorted_keys = np.full_like(keys, -9999)
    sorted_values = np.full_like(values, -9999)

    _logical_warp_pair_merge_sort[1, _THREADS](
        keys,
        values,
        original,
        sorted_keys,
        sorted_values,
    )

    np.testing.assert_array_equal(original, keys)
    for base in range(0, _TOTAL_ITEMS, _LOGICAL_WARP_ITEMS):
        valid_end = base + _LOGICAL_WARP_VALID_ITEMS
        order = np.argsort(keys[base:valid_end])
        np.testing.assert_array_equal(
            sorted_keys[base:valid_end],
            keys[base:valid_end][order],
        )
        np.testing.assert_array_equal(
            sorted_values[base:valid_end],
            values[base:valid_end][order],
        )

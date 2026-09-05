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

import cuda.coop.numba_mlir as coop

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
    _complex_real_greater,
    _less,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _qualified_scalar_group_merge_sort_kernel(source, block_output, warp_output):
    tid = cuda.threadIdx.x
    value = source[tid]
    block_output[tid] = coop.merge_sort_keys(coop.this_block(), value)
    warp_output[tid] = coop.merge_sort_keys(
        coop.this_warp(),
        value,
        descending=True,
    )


def test_qualified_scalar_group_merge_sort_returns_sorted_values():
    values = ((np.arange(THREADS, dtype=np.int32) * 17) % 31) - 15
    block_output = np.zeros_like(values)
    warp_output = np.zeros_like(values)

    _qualified_scalar_group_merge_sort_kernel[1, THREADS](
        values,
        block_output,
        warp_output,
    )

    np.testing.assert_array_equal(block_output, np.sort(values))
    np.testing.assert_array_equal(warp_output, np.sort(values)[::-1])


@cuda.jit
def _qualified_scalar_group_merge_sort_pairs_kernel(
    keys,
    values,
    block_keys,
    block_values,
    warp_keys,
    warp_values,
):
    tid = cuda.threadIdx.x
    block_key, block_value = coop.merge_sort_pairs(
        coop.this_block(),
        keys[tid],
        values[tid],
    )
    warp_key, warp_value = coop.merge_sort_pairs(
        coop.this_warp(),
        keys[tid],
        values[tid],
        descending=True,
    )
    block_keys[tid] = block_key
    block_values[tid] = block_value
    warp_keys[tid] = warp_key
    warp_values[tid] = warp_value


def test_qualified_scalar_group_merge_sort_pairs_preserve_payload_association():
    indices = np.arange(THREADS, dtype=np.int32)
    keys = ((indices * np.int32(17)) % np.int32(THREADS)) - np.int32(16)
    values = (((indices * np.int32(23)) % np.int32(97)) - np.int32(48)).astype(
        np.float32
    )
    block_keys = np.zeros_like(keys)
    block_values = np.zeros_like(values)
    warp_keys = np.zeros_like(keys)
    warp_values = np.zeros_like(values)

    _qualified_scalar_group_merge_sort_pairs_kernel[1, THREADS](
        keys,
        values,
        block_keys,
        block_values,
        warp_keys,
        warp_values,
    )

    ascending_order = np.argsort(keys, kind="stable")
    descending_order = ascending_order[::-1]
    np.testing.assert_array_equal(block_keys, keys[ascending_order])
    np.testing.assert_array_equal(block_values, values[ascending_order])
    np.testing.assert_array_equal(warp_keys, keys[descending_order])
    np.testing.assert_array_equal(warp_values, values[descending_order])


@cuda.jit
def _warp_merge_sort_kernel(
    d_keys, d_values, d_keys_out, d_pair_keys_out, d_values_out
):
    tid = cuda.threadIdx.x
    keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    pair_keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    values = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        keys[i] = d_keys[idx]
        pair_keys[i] = d_keys[idx]
        values[i] = d_values[idx]

    coop._warp.merge_sort_keys(
        keys,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_less,
        threads_in_warp=THREADS,
    )
    coop._warp.merge_sort_pairs(
        pair_keys,
        values,
        keys="int32",
        values="int32",
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_less,
        threads_in_warp=THREADS,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


def test_warp_merge_sort_keys_and_pairs():
    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_values = h_keys + np.int32(1000)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _warp_merge_sort_kernel[1, THREADS](
        h_keys, h_values, h_keys_out, h_pair_keys_out, h_values_out
    )

    expected_keys = np.sort(h_keys)
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_pair_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_keys + np.int32(1000))


def test_warp_merge_sort_keys_value_dtype_two_phase():
    merge_sort = coop._warp.make_merge_sort_keys(
        types.int32,
        ITEMS_PER_THREAD,
        _less,
        value_dtype=types.int32,
        threads_in_warp=THREADS,
    )

    @cuda.jit
    def kernel(d_keys, d_values, d_keys_out, d_values_out):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
        values = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_values.dtype)

        for i in range(ITEMS_PER_THREAD):
            idx = tid * ITEMS_PER_THREAD + i
            keys[i] = d_keys[idx]
            values[i] = d_values[idx]

        merge_sort(keys, values)

        for i in range(ITEMS_PER_THREAD):
            idx = tid * ITEMS_PER_THREAD + i
            d_keys_out[idx] = keys[i]
            d_values_out[idx] = values[i]

    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_values = h_keys * np.int32(7) + np.int32(3)
    h_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    kernel[1, THREADS](h_keys, h_values, h_keys_out, h_values_out)

    expected_pairs = sorted(zip(h_keys, h_values), key=lambda kv: kv[0])
    expected_keys = np.array([key for key, _ in expected_pairs], dtype=np.int32)
    expected_values = np.array([value for _, value in expected_pairs], dtype=np.int32)
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_values)


@cuda.jit
def _warp_merge_sort_thread_data_kernel(
    d_keys, d_values, d_keys_out, d_pair_keys_out, d_values_out
):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
    pair_keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
    values = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_values.dtype)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        keys[i] = d_keys[idx]
        pair_keys[i] = d_keys[idx]
        values[i] = d_values[idx]

    coop._warp.merge_sort_keys(
        keys,
        compare_op=_less,
        threads_in_warp=THREADS,
    )
    coop._warp.merge_sort_pairs(
        pair_keys,
        values,
        compare_op=_less,
        threads_in_warp=THREADS,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


def test_warp_merge_sort_thread_data_infers_items_per_thread():
    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_values = h_keys + np.int32(1000)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _warp_merge_sort_thread_data_kernel[1, THREADS](
        h_keys, h_values, h_keys_out, h_pair_keys_out, h_values_out
    )

    expected_keys = np.sort(h_keys)
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_pair_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_keys + np.int32(1000))


@cuda.jit
def _block_merge_sort_kernel(
    d_keys, d_values, d_keys_out, d_pair_keys_out, d_values_out
):
    tid = cuda.threadIdx.x
    keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    pair_keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    values = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        keys[i] = d_keys[idx]
        pair_keys[i] = d_keys[idx]
        values[i] = d_values[idx]

    coop._block.merge_sort_keys(
        keys,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_less,
    )
    coop._block.merge_sort_pairs(
        pair_keys,
        values,
        keys="int32",
        values="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_less,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


def test_block_merge_sort_keys_and_pairs():
    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_values = h_keys + np.int32(1000)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _block_merge_sort_kernel[1, THREADS](
        h_keys, h_values, h_keys_out, h_pair_keys_out, h_values_out
    )

    expected_keys = np.sort(h_keys)
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_pair_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_keys + np.int32(1000))


def test_block_merge_sort_keys_value_dtype_two_phase():
    merge_sort = coop._block.make_merge_sort_keys(
        types.int32,
        THREADS,
        ITEMS_PER_THREAD,
        _less,
        value_dtype=types.int32,
    )

    @cuda.jit
    def kernel(d_keys, d_values, d_keys_out, d_values_out):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
        values = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_values.dtype)

        for i in range(ITEMS_PER_THREAD):
            idx = tid * ITEMS_PER_THREAD + i
            keys[i] = d_keys[idx]
            values[i] = d_values[idx]

        merge_sort(keys, values)

        for i in range(ITEMS_PER_THREAD):
            idx = tid * ITEMS_PER_THREAD + i
            d_keys_out[idx] = keys[i]
            d_values_out[idx] = values[i]

    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_values = h_keys * np.int32(5) + np.int32(11)
    h_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    kernel[1, THREADS](h_keys, h_values, h_keys_out, h_values_out)

    expected_pairs = sorted(zip(h_keys, h_values), key=lambda kv: kv[0])
    expected_keys = np.array([key for key, _ in expected_pairs], dtype=np.int32)
    expected_values = np.array([value for _, value in expected_pairs], dtype=np.int32)
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_values)


@cuda.jit
def _block_merge_sort_complex_kernel(d_keys, d_out):
    tid = cuda.threadIdx.x
    keys = coop.local.array(ITEMS_PER_THREAD, dtype=d_keys.dtype)

    for i in range(ITEMS_PER_THREAD):
        keys[i] = d_keys[tid * ITEMS_PER_THREAD + i]

    coop._block.merge_sort_keys(
        keys,
        dtype=d_keys.dtype,
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_complex_real_greater,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = keys[i]


def test_block_merge_sort_complex_keys():
    total_items = THREADS * ITEMS_PER_THREAD
    real = ((np.arange(total_items, dtype=np.int32) * 17) % total_items).astype(
        np.float64
    )
    h_keys = (real + 1j * (real + 100.0)).astype(np.complex128)
    h_out = np.zeros_like(h_keys)

    _block_merge_sort_complex_kernel[1, THREADS](h_keys, h_out)

    expected = np.asarray(
        sorted(h_keys, key=lambda value: value.real, reverse=True),
        dtype=np.complex128,
    )
    np.testing.assert_array_equal(h_out, expected)


@cuda.jit
def _block_merge_sort_valid_items_kernel(
    d_keys,
    d_values,
    d_keys_out,
    d_pair_keys_out,
    d_values_out,
    valid_items,
    oob_default,
):
    tid = cuda.threadIdx.x
    keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    pair_keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    values = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        keys[i] = d_keys[idx]
        pair_keys[i] = d_keys[idx]
        values[i] = d_values[idx]

    coop._block.merge_sort_keys(
        keys,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_less,
        valid_items=valid_items,
        oob_default=oob_default,
    )
    coop._block.merge_sort_pairs(
        pair_keys,
        values,
        keys="int32",
        values="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_less,
        valid_items=valid_items,
        oob_default=oob_default,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


def test_block_merge_sort_valid_items_and_oob_default():
    total_items = THREADS * ITEMS_PER_THREAD
    valid_items = np.int32(total_items - 5)
    oob_default = np.int32(9999)
    h_keys = np.arange(total_items, 0, -1, dtype=np.int32)
    h_values = h_keys + np.int32(1000)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _block_merge_sort_valid_items_kernel[1, THREADS](
        h_keys,
        h_values,
        h_keys_out,
        h_pair_keys_out,
        h_values_out,
        valid_items,
        oob_default,
    )

    expected_pairs = sorted(
        zip(h_keys[:valid_items], h_values[:valid_items]),
        key=lambda kv: kv[0],
    )
    expected_keys = np.array([key for key, _ in expected_pairs], dtype=np.int32)
    expected_values = np.array([value for _, value in expected_pairs], dtype=np.int32)
    np.testing.assert_array_equal(h_keys_out[:valid_items], expected_keys)
    np.testing.assert_array_equal(h_pair_keys_out[:valid_items], expected_keys)
    np.testing.assert_array_equal(h_values_out[:valid_items], expected_values)

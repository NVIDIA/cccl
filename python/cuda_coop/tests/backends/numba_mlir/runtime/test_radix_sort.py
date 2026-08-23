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
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _qualified_scalar_group_radix_sort_kernel(
    source,
    ascending_output,
    descending_output,
):
    tid = cuda.threadIdx.x
    value = source[tid]
    ascending_output[tid] = coop.radix_sort_keys(coop.this_block(), value)
    descending_output[tid] = coop.radix_sort_keys(
        coop.this_block(),
        value,
        descending=True,
    )


def test_qualified_scalar_group_radix_sort_returns_sorted_values():
    values = ((np.arange(THREADS, dtype=np.int32) * 19) % 37) - 18
    ascending_output = np.zeros_like(values)
    descending_output = np.zeros_like(values)

    _qualified_scalar_group_radix_sort_kernel[1, THREADS](
        values,
        ascending_output,
        descending_output,
    )

    np.testing.assert_array_equal(ascending_output, np.sort(values))
    np.testing.assert_array_equal(descending_output, np.sort(values)[::-1])


@cuda.jit
def _qualified_scalar_group_radix_sort_pairs_kernel(
    keys,
    values,
    key_output,
    value_output,
):
    tid = cuda.threadIdx.x
    key, value = coop.radix_sort_pairs(
        coop.this_block(),
        keys[tid],
        values[tid],
        begin_bit=4,
        end_bit=9,
    )
    key_output[tid] = key
    value_output[tid] = value


def test_qualified_scalar_group_radix_sort_pairs_use_selected_bits_and_payload():
    indices = np.arange(THREADS, dtype=np.uint32)
    digits = (indices * np.uint32(13) + np.uint32(5)) % np.uint32(THREADS)
    high_bits = (np.uint32(THREADS - 1) - indices) << np.uint32(9)
    low_bits = (indices * np.uint32(3)) & np.uint32(0xF)
    keys = high_bits | (digits << np.uint32(4)) | low_bits
    values = (((indices * np.uint32(29)) % np.uint32(101)).astype(np.int32)) - 50
    key_output = np.zeros_like(keys)
    value_output = np.zeros_like(values)

    _qualified_scalar_group_radix_sort_pairs_kernel[1, THREADS](
        keys,
        values,
        key_output,
        value_output,
    )

    selected_digits = (keys >> np.uint32(4)) & np.uint32(0x1F)
    expected_order = np.argsort(selected_digits, kind="stable")
    np.testing.assert_array_equal(key_output, keys[expected_order])
    np.testing.assert_array_equal(value_output, values[expected_order])


@cuda.jit
def _block_radix_sort_kernel(
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

    coop._block.radix_sort_keys(
        keys,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    coop._block.radix_sort_pairs(
        pair_keys,
        values,
        key_dtype="int32",
        value_dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


@cuda.jit
def _block_radix_sort_descending_kernel(
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

    coop._block.radix_sort_keys_descending(
        keys,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    coop._block.radix_sort_pairs_descending(
        pair_keys,
        values,
        key_dtype="int32",
        value_dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


def test_block_radix_sort_keys_and_pairs():
    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_values = h_keys + np.int32(1000)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _block_radix_sort_kernel[1, THREADS](
        h_keys, h_values, h_keys_out, h_pair_keys_out, h_values_out
    )

    expected_keys = np.sort(h_keys)
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_pair_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_keys + np.int32(1000))


@cuda.jit
def _block_radix_sort_thread_data_kernel(
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

    coop._block.radix_sort_keys(keys, threads_per_block=THREADS)
    coop._block.radix_sort_pairs(pair_keys, values, threads_per_block=THREADS)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


def test_block_radix_sort_thread_data_infers_items_per_thread():
    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_values = h_keys + np.int32(1000)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _block_radix_sort_thread_data_kernel[1, THREADS](
        h_keys, h_values, h_keys_out, h_pair_keys_out, h_values_out
    )

    expected_keys = np.sort(h_keys)
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_pair_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_keys + np.int32(1000))


def test_block_radix_sort_descending_keys_and_pairs():
    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_values = h_keys + np.int32(1000)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _block_radix_sort_descending_kernel[1, THREADS](
        h_keys, h_values, h_keys_out, h_pair_keys_out, h_values_out
    )

    expected_keys = np.sort(h_keys)[::-1]
    np.testing.assert_array_equal(h_keys_out, expected_keys)
    np.testing.assert_array_equal(h_pair_keys_out, expected_keys)
    np.testing.assert_array_equal(h_values_out, expected_keys + np.int32(1000))


def test_block_radix_sort_two_phase_bits_temp_storage():
    radix_sort = coop._block.make_radix_sort_keys(
        types.uint32,
        THREADS,
        ITEMS_PER_THREAD,
        begin_bit=0,
        end_bit=4,
    )
    temp_storage_bytes = int(radix_sort.temp_storage_bytes)
    temp_storage_alignment = int(radix_sort.temp_storage_alignment)

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
        temp_storage = coop.TempStorage(temp_storage_bytes, temp_storage_alignment)

        for i in range(ITEMS_PER_THREAD):
            keys[i] = d_in[tid * ITEMS_PER_THREAD + i]

        radix_sort(keys, 0, 4, temp_storage=temp_storage)

        for i in range(ITEMS_PER_THREAD):
            d_out[tid * ITEMS_PER_THREAD + i] = keys[i]

    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.uint32)) & 0xF
    h_output = np.zeros_like(h_input)

    kernel[1, THREADS](h_input, h_output)

    expected = np.sort(h_input)
    np.testing.assert_array_equal(h_output, expected)


def test_block_radix_sort_keys_value_dtype_two_phase():
    radix_sort = coop._block.make_radix_sort_keys(
        types.int32,
        THREADS,
        ITEMS_PER_THREAD,
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

        radix_sort(keys, values)

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


def test_block_radix_sort_blocked_to_striped_two_phase():
    radix_sort = coop._block.make_radix_sort_keys(
        types.int32,
        THREADS,
        ITEMS_PER_THREAD,
        blocked_to_striped=True,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

        for i in range(ITEMS_PER_THREAD):
            keys[i] = d_in[tid * ITEMS_PER_THREAD + i]

        radix_sort(keys)

        for i in range(ITEMS_PER_THREAD):
            d_out[tid * ITEMS_PER_THREAD + i] = keys[i]

    h_input = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    kernel[1, THREADS](h_input, h_output)

    sorted_keys = np.sort(h_input)
    expected = np.empty_like(sorted_keys)
    for tid in range(THREADS):
        for i in range(ITEMS_PER_THREAD):
            expected[tid * ITEMS_PER_THREAD + i] = sorted_keys[tid + i * THREADS]
    np.testing.assert_array_equal(h_output, expected)


@cuda.jit
def _block_radix_sort_blocked_to_striped_single_phase_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    for i in range(ITEMS_PER_THREAD):
        keys[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.radix_sort_keys(
        keys,
        threads_per_block=THREADS,
        blocked_to_striped=True,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = keys[i]


def test_block_radix_sort_blocked_to_striped_single_phase():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_radix_sort_blocked_to_striped_single_phase_kernel[1, THREADS](
        h_input, h_output
    )

    sorted_keys = np.sort(h_input)
    expected = np.empty_like(sorted_keys)
    for tid in range(THREADS):
        for i in range(ITEMS_PER_THREAD):
            expected[tid * ITEMS_PER_THREAD + i] = sorted_keys[tid + i * THREADS]
    np.testing.assert_array_equal(h_output, expected)

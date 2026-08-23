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
def _block_topk_kernel(d_keys, d_values, d_keys_out, d_pair_keys_out, d_values_out, k):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
    pair_keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
    values = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_values.dtype)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        keys[i] = d_keys[idx]
        pair_keys[i] = d_keys[idx]
        values[i] = d_values[idx]

    coop._block.topk_max_keys(keys, k, threads_per_block=THREADS)
    coop._block.topk_min_pairs(pair_keys, values, k, threads_per_block=THREADS)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_keys_out[idx] = keys[i]
        d_pair_keys_out[idx] = pair_keys[i]
        d_values_out[idx] = values[i]


def test_block_topk_keys_and_pairs():
    k = np.int32(7)
    total_items = THREADS * ITEMS_PER_THREAD
    h_keys = ((np.arange(total_items, dtype=np.int32) * 29) % total_items).astype(
        np.int32
    )
    h_values = h_keys * np.int32(11) + np.int32(3)
    h_keys_out = np.zeros_like(h_keys)
    h_pair_keys_out = np.zeros_like(h_keys)
    h_values_out = np.zeros_like(h_values)

    _block_topk_kernel[1, THREADS](
        h_keys, h_values, h_keys_out, h_pair_keys_out, h_values_out, k
    )

    actual_keys = np.sort(h_keys_out[:k])
    expected_keys = np.sort(h_keys)[-k:]
    np.testing.assert_array_equal(actual_keys, expected_keys)

    actual_pairs = sorted(zip(h_pair_keys_out[:k], h_values_out[:k], strict=True))
    expected_pair_keys = np.sort(h_keys)[:k]
    expected_pairs = sorted((key, key * 11 + 3) for key in expected_pair_keys)
    assert actual_pairs == expected_pairs


def test_block_topk_two_phase_factory_num_valid_constant():
    valid_items = np.int32(THREADS * ITEMS_PER_THREAD - 5)
    k = np.int32(11)
    topk = coop._block.make_topk_min_keys(
        types.int32,
        THREADS,
        ITEMS_PER_THREAD,
        num_valid=valid_items,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
        for i in range(ITEMS_PER_THREAD):
            keys[i] = d_in[tid * ITEMS_PER_THREAD + i]

        topk(keys, k)

        for i in range(ITEMS_PER_THREAD):
            d_out[tid * ITEMS_PER_THREAD + i] = keys[i]

    total_items = THREADS * ITEMS_PER_THREAD
    h_input = np.arange(total_items, 0, -1, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    kernel[1, THREADS](h_input, h_output)

    actual = np.sort(h_output[:k])
    expected = np.sort(h_input[:valid_items])[:k]
    np.testing.assert_array_equal(actual, expected)


@cuda.jit
def _block_topk_runtime_bits_kernel(d_in, d_out, k, begin_bit, end_bit):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    for i in range(ITEMS_PER_THREAD):
        keys[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.topk_max_keys(
        keys,
        k,
        threads_per_block=THREADS,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = keys[i]


def test_block_topk_runtime_bit_range():
    k = np.int32(8)
    begin_bit = np.int32(0)
    end_bit = np.int32(4)
    total_items = THREADS * ITEMS_PER_THREAD
    indices = np.arange(total_items, dtype=np.uint32)
    h_input = (indices << np.uint32(4)) | ((indices * np.uint32(7)) & np.uint32(0xF))
    h_output = np.zeros_like(h_input)

    _block_topk_runtime_bits_kernel[1, THREADS](
        h_input, h_output, k, begin_bit, end_bit
    )

    actual_digits = np.sort(h_output[:k] & np.uint32(0xF))
    expected_digits = np.sort(h_input & np.uint32(0xF))[-k:]
    np.testing.assert_array_equal(actual_digits, expected_digits)

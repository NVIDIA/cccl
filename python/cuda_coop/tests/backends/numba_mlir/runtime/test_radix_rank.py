# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")


import cuda.coop.numba_mlir as coop

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
    _exclusive_digit_prefix_reference,
    _validate_ranks,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _block_radix_rank_kernel(d_in, d_ranks):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.uint32)
    ranks = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.radix_rank(
        items,
        ranks,
        dtype="uint32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        begin_bit=0,
        end_bit=4,
    )

    for i in range(ITEMS_PER_THREAD):
        d_ranks[tid * ITEMS_PER_THREAD + i] = ranks[i]


def test_block_radix_rank_assigns_digit_ranges():
    h_input = (
        np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.uint32) * np.uint32(7)
    ) & np.uint32(0xF)
    h_ranks = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.int32)

    _block_radix_rank_kernel[1, THREADS](h_input, h_ranks)

    _validate_ranks(h_input, h_ranks, begin_bit=0, end_bit=4)


@cuda.jit
def _block_radix_rank_exclusive_digit_prefix_kernel(d_in, d_ranks, d_prefix):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.uint32)
    ranks = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    exclusive_digit_prefix = cuda.local.array(1, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.radix_rank(
        items,
        ranks,
        dtype="uint32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        begin_bit=0,
        end_bit=4,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )

    for i in range(ITEMS_PER_THREAD):
        d_ranks[tid * ITEMS_PER_THREAD + i] = ranks[i]
    d_prefix[tid] = exclusive_digit_prefix[0]


def test_block_radix_rank_exclusive_digit_prefix():
    h_input = (
        np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.uint32) * np.uint32(7)
    ) & np.uint32(0xF)
    h_ranks = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_prefix = np.zeros(THREADS, dtype=np.int32)

    _block_radix_rank_exclusive_digit_prefix_kernel[1, THREADS](
        h_input, h_ranks, h_prefix
    )

    _validate_ranks(h_input, h_ranks, begin_bit=0, end_bit=4)
    expected = _exclusive_digit_prefix_reference(h_input, begin_bit=0, end_bit=4)
    valid_mask = expected[:, 0] != -1
    np.testing.assert_array_equal(h_prefix[valid_mask], expected[:, 0][valid_mask])

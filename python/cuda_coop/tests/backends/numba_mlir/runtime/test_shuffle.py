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
from cuda.coop.numba_mlir._block import (
    BlockShuffleType,
)

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _block_shuffle_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    value = d_in[tid]
    shuffled = coop._block.shuffle(
        value,
        block_shuffle_type=BlockShuffleType.Offset,
        dtype="int32",
        threads_per_block=THREADS,
        distance=1,
        block_suffix=None,
    )

    d_out[tid] = -1
    if tid + 1 < THREADS:
        d_out[tid] = shuffled


def test_block_shuffle_offset_scalar():
    h_input = np.arange(THREADS, dtype=np.int32)
    h_output = np.full(THREADS, -1, dtype=np.int32)

    _block_shuffle_kernel[1, THREADS](h_input, h_output)

    expected = np.full_like(h_output, -1)
    expected[:-1] = h_input[1:]
    np.testing.assert_array_equal(h_output, expected)


@cuda.jit
def _block_shuffle_up_scalar_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    value = d_in[tid]
    shuffled = coop._block.shuffle(
        value,
        block_shuffle_type=BlockShuffleType.Up,
        dtype="int32",
        threads_per_block=THREADS,
        distance=2,
    )

    d_out[tid] = value
    if tid >= 2:
        d_out[tid] = shuffled


def test_block_shuffle_up_scalar():
    h_input = np.arange(THREADS, dtype=np.int32)
    h_output = np.empty_like(h_input)

    _block_shuffle_up_scalar_kernel[1, THREADS](h_input, h_output)

    expected = h_input.copy()
    expected[2:] = h_input[:-2]
    np.testing.assert_array_equal(h_output, expected)


@cuda.jit
def _block_shuffle_up_array_kernel(d_in, d_out, d_suffix):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    block_suffix = cuda.local.array(1, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.shuffle(
        items,
        items,
        block_shuffle_type=BlockShuffleType.Up,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        block_suffix=block_suffix,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = items[i]
    d_suffix[tid] = block_suffix[0]


def test_block_shuffle_up_array_with_suffix():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.empty_like(h_input)
    h_suffix = np.empty(THREADS, dtype=np.int32)

    _block_shuffle_up_array_kernel[1, THREADS](h_input, h_output, h_suffix)

    expected = h_input.copy()
    expected[1:] = h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)
    np.testing.assert_array_equal(h_suffix, np.full(THREADS, h_input[-1]))

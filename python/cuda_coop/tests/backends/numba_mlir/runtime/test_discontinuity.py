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
from cuda.coop.numba_mlir._block import (
    BlockDiscontinuityType,
)

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
    _different,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _block_discontinuity_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    flags = cuda.local.array(ITEMS_PER_THREAD, cuda.boolean)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.discontinuity(
        items,
        flags,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        flag_op=_different,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = flags[i]


@cuda.jit(device=True)
def _different_for_temp_storage(a, b):
    return a != b


@cuda.jit
def _block_discontinuity_thread_data_temp_storage_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    temp_storage = coop.TempStorage()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    flags = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.boolean)

    coop._block.load(d_in, items)
    coop._block.discontinuity(
        items,
        flags,
        flag_op=_different_for_temp_storage,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
        temp_storage=temp_storage,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = flags[i]


@cuda.jit
def _block_discontinuity_tails_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    flags = cuda.local.array(ITEMS_PER_THREAD, cuda.boolean)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.discontinuity(
        items,
        flags,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        flag_op=_different,
        block_discontinuity_type=BlockDiscontinuityType.TAILS,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = flags[i]


@cuda.jit
def _block_discontinuity_heads_and_tails_kernel(d_in, d_heads, d_tails):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    head_flags = cuda.local.array(ITEMS_PER_THREAD, cuda.boolean)
    tail_flags = cuda.local.array(ITEMS_PER_THREAD, cuda.boolean)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.discontinuity(
        items,
        head_flags,
        tail_flags,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        flag_op=_different,
        block_discontinuity_type=BlockDiscontinuityType.HEADS_AND_TAILS,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_heads[idx] = head_flags[i]
        d_tails[idx] = tail_flags[i]


@cuda.jit
def _block_discontinuity_heads_and_tails_boundary_kernel(
    d_in, d_heads, d_tails, tile_predecessor, tile_successor
):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    head_flags = cuda.local.array(ITEMS_PER_THREAD, cuda.boolean)
    tail_flags = cuda.local.array(ITEMS_PER_THREAD, cuda.boolean)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.discontinuity(
        items,
        head_flags,
        tail_flags,
        tile_predecessor,
        tile_successor,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        flag_op=_different,
        block_discontinuity_type=BlockDiscontinuityType.HEADS_AND_TAILS,
        tile_predecessor_item=True,
        tile_successor_item=True,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_heads[idx] = head_flags[i]
        d_tails[idx] = tail_flags[i]


@cuda.jit
def _block_discontinuity_heads_tile_kernel(d_in, d_out, tile_predecessor):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    flags = cuda.local.array(ITEMS_PER_THREAD, cuda.boolean)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.discontinuity(
        items,
        flags,
        tile_predecessor,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        flag_op=_different,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
        tile_predecessor_item=True,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = flags[i]


def test_block_discontinuity_flag_heads():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) // 3) % 5
    h_output = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)

    _block_discontinuity_kernel[1, THREADS](h_input, h_output)

    expected = np.empty_like(h_output)
    expected[0] = True
    expected[1:] = h_input[1:] != h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)


def test_block_discontinuity_thread_data_temp_storage_flag_heads():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) // 3) % 5
    h_output = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)

    _block_discontinuity_thread_data_temp_storage_kernel[1, THREADS](h_input, h_output)

    expected = np.empty_like(h_output)
    expected[0] = True
    expected[1:] = h_input[1:] != h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)


def test_block_discontinuity_flag_tails():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) // 3) % 5
    h_output = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)

    _block_discontinuity_tails_kernel[1, THREADS](h_input, h_output)

    expected = np.empty_like(h_output)
    expected[-1] = True
    expected[:-1] = h_input[1:] != h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)


def test_block_discontinuity_flag_heads_and_tails():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) // 3) % 5
    h_heads = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)
    h_tails = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)

    _block_discontinuity_heads_and_tails_kernel[1, THREADS](h_input, h_heads, h_tails)

    expected_heads = np.empty_like(h_heads)
    expected_heads[0] = True
    expected_heads[1:] = h_input[1:] != h_input[:-1]
    expected_tails = np.empty_like(h_tails)
    expected_tails[-1] = True
    expected_tails[:-1] = h_input[1:] != h_input[:-1]
    np.testing.assert_array_equal(h_heads, expected_heads)
    np.testing.assert_array_equal(h_tails, expected_tails)


def test_block_discontinuity_flag_heads_and_tails_boundaries():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) // 3) % 5
    h_heads = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)
    h_tails = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)
    tile_predecessor = np.int32(5)
    tile_successor = np.int32(7)

    _block_discontinuity_heads_and_tails_boundary_kernel[1, THREADS](
        h_input,
        h_heads,
        h_tails,
        tile_predecessor,
        tile_successor,
    )

    expected_heads = np.empty_like(h_heads)
    expected_heads[0] = h_input[0] != tile_predecessor
    expected_heads[1:] = h_input[1:] != h_input[:-1]
    expected_tails = np.empty_like(h_tails)
    expected_tails[-1] = h_input[-1] != tile_successor
    expected_tails[:-1] = h_input[1:] != h_input[:-1]
    np.testing.assert_array_equal(h_heads, expected_heads)
    np.testing.assert_array_equal(h_tails, expected_tails)


def test_block_discontinuity_flag_heads_tile_predecessor():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) // 3) % 5
    h_output = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.bool_)
    tile_predecessor = np.int32(5)

    _block_discontinuity_heads_tile_kernel[1, THREADS](
        h_input, h_output, tile_predecessor
    )

    expected = np.empty_like(h_output)
    expected[0] = h_input[0] != tile_predecessor
    expected[1:] = h_input[1:] != h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)

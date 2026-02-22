# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
from helpers import row_major_tid
from numba import cuda

from cuda import coop
from cuda.coop.block import BlockDiscontinuityType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit(device=True)
def flag_op(lhs, rhs):
    return numba.int32(1 if lhs != rhs else 0)


def test_block_discontinuity_flag_heads_thread_data_temp_storage():
    dtype = np.int32
    flag_dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    block_disc = coop.block.discontinuity(
        dtype,
        threads_per_block,
        items_per_thread,
        flag_op=flag_op,
        flag_dtype=flag_dtype,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
    )
    temp_storage_bytes = block_disc.temp_storage_bytes
    temp_storage_alignment = block_disc.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        flags = coop.ThreadData(items_per_thread, dtype=flag_dtype)

        coop.block.load(d_in, items)
        coop.block.discontinuity(
            items,
            flags,
            flag_op=flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS,
            temp_storage=temp_storage,
        )

        tid = row_major_tid()
        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = flags[i]

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    num_items = num_threads * items_per_thread

    h_input = np.random.randint(0, 4, num_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_items, dtype=flag_dtype)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()

    h_reference = np.zeros_like(h_output)
    for idx in range(num_items):
        if idx == 0:
            h_reference[idx] = 1
        else:
            h_reference[idx] = 1 if h_input[idx] != h_input[idx - 1] else 0

    np.testing.assert_array_equal(h_output, h_reference)


def test_block_discontinuity_two_phase_heads():
    dtype = np.int32
    flag_dtype = np.int32
    threads_per_block = 64
    items_per_thread = 2

    block_disc = coop.block.discontinuity(
        dtype,
        threads_per_block,
        items_per_thread,
        flag_op=flag_op,
        flag_dtype=flag_dtype,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
    )
    temp_storage_bytes = block_disc.temp_storage_bytes
    temp_storage_alignment = block_disc.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        flags = coop.ThreadData(items_per_thread, dtype=flag_dtype)

        coop.block.load(d_in, items)
        block_disc(
            items,
            flags,
            flag_op=flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS,
            temp_storage=temp_storage,
        )

        tid = row_major_tid()
        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = flags[i]

    num_threads = threads_per_block
    num_items = num_threads * items_per_thread

    h_input = np.random.randint(0, 4, num_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_items, dtype=flag_dtype)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()

    h_reference = np.zeros_like(h_output)
    for idx in range(num_items):
        if idx == 0:
            h_reference[idx] = 1
        else:
            h_reference[idx] = 1 if h_input[idx] != h_input[idx - 1] else 0

    np.testing.assert_array_equal(h_output, h_reference)


def test_block_discontinuity_heads_tile_predecessor():
    dtype = np.int32
    flag_dtype = np.int32
    threads_per_block = 64
    items_per_thread = 2
    tile_predecessor_item = np.int32(5)

    block_disc = coop.block.discontinuity(
        dtype,
        threads_per_block,
        items_per_thread,
        flag_op=flag_op,
        flag_dtype=flag_dtype,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
    )
    temp_storage_bytes = block_disc.temp_storage_bytes
    temp_storage_alignment = block_disc.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        flags = coop.ThreadData(items_per_thread, dtype=flag_dtype)

        coop.block.load(d_in, items)
        coop.block.discontinuity(
            items,
            flags,
            flag_op=flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS,
            tile_predecessor_item=tile_predecessor_item,
            temp_storage=temp_storage,
        )

        tid = row_major_tid()
        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = flags[i]

    num_threads = threads_per_block
    num_items = num_threads * items_per_thread

    h_input = np.random.randint(0, 4, num_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_items, dtype=flag_dtype)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()

    h_reference = np.zeros_like(h_output)
    prev = tile_predecessor_item
    for idx in range(num_items):
        h_reference[idx] = 1 if h_input[idx] != prev else 0
        prev = h_input[idx]

    np.testing.assert_array_equal(h_output, h_reference)


def test_block_discontinuity_heads_and_tails_tile_pred_succ():
    dtype = np.int32
    flag_dtype = np.int32
    threads_per_block = 64
    items_per_thread = 2
    tile_predecessor_item = np.int32(0)
    tile_successor_item = np.int32(9)

    block_disc = coop.block.discontinuity(
        dtype,
        threads_per_block,
        items_per_thread,
        flag_op=flag_op,
        flag_dtype=flag_dtype,
        block_discontinuity_type=BlockDiscontinuityType.HEADS_AND_TAILS,
    )
    temp_storage_bytes = block_disc.temp_storage_bytes
    temp_storage_alignment = block_disc.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_head_out, d_tail_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        head_flags = coop.ThreadData(items_per_thread, dtype=flag_dtype)
        tail_flags = coop.ThreadData(items_per_thread, dtype=flag_dtype)

        coop.block.load(d_in, items)
        coop.block.discontinuity(
            items,
            head_flags,
            tail_flags,
            flag_op=flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS_AND_TAILS,
            tile_predecessor_item=tile_predecessor_item,
            tile_successor_item=tile_successor_item,
            temp_storage=temp_storage,
        )

        tid = row_major_tid()
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            d_head_out[idx] = head_flags[i]
            d_tail_out[idx] = tail_flags[i]

    num_threads = threads_per_block
    num_items = num_threads * items_per_thread

    h_input = np.random.randint(0, 4, num_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_head = cuda.device_array(num_items, dtype=flag_dtype)
    d_tail = cuda.device_array(num_items, dtype=flag_dtype)

    kernel[1, threads_per_block](d_input, d_head, d_tail)
    cuda.synchronize()

    h_head = d_head.copy_to_host()
    h_tail = d_tail.copy_to_host()

    ref_head = np.zeros_like(h_head)
    ref_tail = np.zeros_like(h_tail)
    prev = tile_predecessor_item
    for idx in range(num_items):
        ref_head[idx] = 1 if h_input[idx] != prev else 0
        prev = h_input[idx]
    for idx in range(num_items):
        nxt = tile_successor_item if idx == num_items - 1 else h_input[idx + 1]
        ref_tail[idx] = 1 if h_input[idx] != nxt else 0

    np.testing.assert_array_equal(h_head, ref_head)
    np.testing.assert_array_equal(h_tail, ref_tail)

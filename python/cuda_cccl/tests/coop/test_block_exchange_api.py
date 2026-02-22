# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop
from cuda.coop.block import BlockExchangeType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-end imports


def test_block_exchange_striped_to_blocked():
    # example-begin striped-to-blocked
    threads_per_block = 32
    items_per_thread = 2
    num_threads = threads_per_block
    total_items = num_threads * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(items_per_thread, dtype=numba.int32)

        for i in range(items_per_thread):
            items[i] = d_in[tid + i * num_threads]

        coop.block.exchange(
            items,
            items_per_thread=items_per_thread,
            block_exchange_type=BlockExchangeType.StripedToBlocked,
        )

        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = items[i]

    # example-end striped-to-blocked

    h_input = np.empty(total_items, dtype=np.int32)
    for i in range(total_items):
        tid = i % num_threads
        item = i // num_threads
        h_input[i] = tid * 100 + item

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.empty_like(h_output)
    for blocked_idx in range(total_items):
        s_tid = blocked_idx % num_threads
        s_item_idx_in_stripe = blocked_idx // num_threads
        src_striped_1d_idx = s_tid + s_item_idx_in_stripe * num_threads
        expected[blocked_idx] = h_input[src_striped_1d_idx]

    np.testing.assert_array_equal(h_output, expected)


def test_block_exchange_blocked_to_striped():
    # example-begin blocked-to-striped
    threads_per_block = 32
    items_per_thread = 2
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(items_per_thread, dtype=numba.int32)

        for i in range(items_per_thread):
            items[i] = d_in[tid * items_per_thread + i]

        coop.block.exchange(
            items,
            items_per_thread=items_per_thread,
            block_exchange_type=BlockExchangeType.BlockedToStriped,
        )

        for i in range(items_per_thread):
            d_out[tid + i * threads_per_block] = items[i]

    # example-end blocked-to-striped

    h_input = np.empty(total_items, dtype=np.int32)
    for i in range(total_items):
        h_input[i] = i

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.empty_like(h_output)
    for tid in range(threads_per_block):
        for item in range(items_per_thread):
            expected[tid + item * threads_per_block] = h_input[
                tid + item * threads_per_block
            ]

    np.testing.assert_array_equal(h_output, expected)

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_warp_exchange_striped_to_blocked():
    threads_in_warp = 32
    items_per_thread = 4

    warp_exchange = coop.warp.exchange(
        numba.int32,
        items_per_thread,
        threads_in_warp=threads_in_warp,
        warp_exchange_type=coop.warp.WarpExchangeType.StripedToBlocked,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        input_items = cuda.local.array(items_per_thread, numba.int32)
        output_items = cuda.local.array(items_per_thread, numba.int32)

        for i in range(items_per_thread):
            input_items[i] = d_in[tid + i * threads_in_warp]

        warp_exchange(input_items, output_items)

        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = output_items[i]

    total_items = threads_in_warp * items_per_thread
    h_input = np.random.randint(0, 64, total_items, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    h_output = d_output.copy_to_host()

    h_reference = np.zeros_like(h_output)
    for blocked_idx in range(total_items):
        source_thread = blocked_idx % threads_in_warp
        source_item = blocked_idx // threads_in_warp
        source_idx = source_thread + source_item * threads_in_warp
        h_reference[blocked_idx] = h_input[source_idx]

    np.testing.assert_array_equal(h_output, h_reference)


def test_warp_exchange_scatter_to_striped():
    threads_in_warp = 16
    items_per_thread = 4
    total_items = threads_in_warp * items_per_thread

    warp_exchange = coop.warp.exchange(
        numba.int32,
        items_per_thread,
        threads_in_warp=threads_in_warp,
        warp_exchange_type=coop.warp.WarpExchangeType.ScatterToStriped,
        offset_dtype=numba.int32,
    )

    @cuda.jit
    def kernel(d_in, d_ranks, d_out):
        tid = cuda.threadIdx.x
        input_items = cuda.local.array(items_per_thread, numba.int32)
        ranks = cuda.local.array(items_per_thread, numba.int32)
        output_items = cuda.local.array(items_per_thread, numba.int32)

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            input_items[i] = d_in[idx]
            ranks[i] = d_ranks[idx]

        warp_exchange(input_items, output_items, ranks)

        for i in range(items_per_thread):
            d_out[tid + i * threads_in_warp] = output_items[i]

    h_input = np.arange(total_items, dtype=np.int32)
    h_ranks = np.arange(total_items - 1, -1, -1, dtype=np.int32)
    h_expected = np.empty_like(h_input)
    for idx, rank in enumerate(h_ranks):
        h_expected[rank] = h_input[idx]

    d_input = cuda.to_device(h_input)
    d_ranks = cuda.to_device(h_ranks)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_in_warp](d_input, d_ranks, d_output)
    h_output = d_output.copy_to_host()

    np.testing.assert_array_equal(h_output, h_expected)

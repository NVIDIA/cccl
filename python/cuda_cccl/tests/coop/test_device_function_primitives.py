# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop
from cuda.coop import BlockLoadAlgorithm, BlockStoreAlgorithm

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_device_function_single_phase_block_and_warp_primitives():
    threads_per_block = 64
    warps_per_block = threads_per_block // 32

    @cuda.jit(device=True)
    def device_block_sum(val):
        return coop.block.sum(val, items_per_thread=1)

    @cuda.jit(device=True)
    def device_warp_sum(val):
        return coop.warp.sum(val)

    @cuda.jit
    def kernel(d_in, d_out_block, d_out_warp):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_sum = device_warp_sum(val)
        block_sum = device_block_sum(val)
        if tid == 0:
            d_out_block[0] = block_sum
        if tid % 32 == 0:
            d_out_warp[tid // 32] = warp_sum

    h_input = np.random.randint(0, 64, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_block = cuda.device_array(1, dtype=np.int32)
    d_out_warp = cuda.device_array(warps_per_block, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_out_block, d_out_warp)
    cuda.synchronize()

    h_out_block = d_out_block.copy_to_host()
    h_out_warp = d_out_warp.copy_to_host()

    expected_block = np.sum(h_input)
    expected_warp = np.array(
        [np.sum(h_input[i * 32 : (i + 1) * 32]) for i in range(warps_per_block)],
        dtype=np.int32,
    )

    assert h_out_block[0] == expected_block
    assert np.array_equal(h_out_warp, expected_warp)


def test_device_function_histogram_parent_child_with_kernel_block_primitives():
    threads_per_block = 64
    items_per_thread = 1
    bins = 8

    @cuda.jit(device=True)
    def device_histogram(d_in, d_hist):
        tid = cuda.threadIdx.x
        smem_histogram = coop.shared.array(bins, dtype=np.uint32)
        thread_samples = coop.local.array(items_per_thread, dtype=np.int32)
        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()
        coop.block.load(
            d_in,
            thread_samples,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
        )
        histo.composite(thread_samples)
        cuda.syncthreads()
        if tid < bins:
            d_hist[tid] = smem_histogram[tid]

    @cuda.jit
    def kernel(d_in, d_hist, d_out):
        thread_samples = cuda.local.array(items_per_thread, numba.int32)

        coop.block.load(
            d_in,
            thread_samples,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
        )

        coop.block.store(
            d_out,
            thread_samples,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.DIRECT,
        )

        device_histogram(d_in, d_hist)

    h_input = np.random.randint(0, bins, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_hist = cuda.device_array(bins, dtype=np.uint32)
    d_out = cuda.device_array(threads_per_block, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_hist, d_out)
    cuda.synchronize()

    h_hist = d_hist.copy_to_host()
    h_out = d_out.copy_to_host()

    expected_hist = np.bincount(h_input, minlength=bins).astype(np.uint32)

    np.testing.assert_array_equal(h_out, h_input)
    np.testing.assert_array_equal(h_hist, expected_hist)

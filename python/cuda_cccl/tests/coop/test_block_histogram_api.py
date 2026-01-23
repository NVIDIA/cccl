# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-end imports


def test_block_histogram_init_composite():
    # example-begin histogram
    threads_per_block = 128
    items_per_thread = 1
    bins = 4

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        smem_histogram = cuda.shared.array(bins, numba.int32)
        thread_samples = cuda.local.array(items_per_thread, numba.int32)

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()

        thread_samples[0] = d_in[tid]
        histo.composite(thread_samples)
        cuda.syncthreads()

        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    # example-end histogram

    h_input = np.arange(threads_per_block, dtype=np.int32) % bins
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.zeros(bins, dtype=np.int32)
    for value in h_input:
        expected[value] += 1

    np.testing.assert_array_equal(h_output, expected)

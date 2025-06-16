# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda
from pynvjitlink import patch

import cuda.cccl.cooperative.experimental as cudax

patch.patch_numba_linker(lto=True)
# example-end imports

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_reduction():
    # example-begin reduce
    def op(a, b):
        return a if a > b else b

    threads_per_block = 128
    block_reduce = cudax.block.reduce(numba.int32, threads_per_block, op)

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        block_output = block_reduce(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # example-end reduce

    h_input = np.random.randint(0, 42, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    h_expected = np.max(h_input)

    assert h_output[0] == h_expected


def test_block_sum():
    # example-begin sum
    threads_per_block = 128
    block_sum = cudax.block.sum(numba.int32, threads_per_block)

    @cuda.jit(link=block_sum.files)
    def kernel(input, output):
        block_output = block_sum(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # example-end sum

    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_per_block

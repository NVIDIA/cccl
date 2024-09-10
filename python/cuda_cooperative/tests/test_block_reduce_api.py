# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import numba
from numba import cuda
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-begin imports
import cuda.cooperative.experimental as cudax
from pynvjitlink import patch
patch.patch_numba_linker(lto=True)
# example-end imports


def test_block_reduction():
    def op(a, b):
        return a if a > b else b

    # example-begin reduce
    threads_in_block = 128
    block_reduce = cudax.block.reduce(numba.int32, threads_in_block, op)
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        block_output = block_reduce(temp_storage, input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = block_output
    # example-end reduce

    h_input = np.random.randint(0, 42, threads_in_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_in_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    h_expected = np.max(h_input)

    assert h_output[0] == h_expected


def test_block_sum():
    # example-begin sum
    threads_in_block = 128
    block_sum = cudax.block.sum(numba.int32, threads_in_block)
    temp_storage_bytes = block_sum.temp_storage_bytes

    @cuda.jit(link=block_sum.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        block_output = block_sum(temp_storage, input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = block_output
    # example-end sum

    h_input = np.ones(threads_in_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_in_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_in_block

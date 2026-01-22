# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-end imports


def test_block_reduction():
    # example-begin reduce
    @cuda.jit(device=True)
    def op(a, b):
        return a if a > b else b

    threads_per_block = 128

    @cuda.jit
    def kernel(input, output):
        block_output = coop.block.reduce(
            input[cuda.threadIdx.x],
            items_per_thread=1,
            binary_op=op,
        )

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

    @cuda.jit
    def kernel(input, output):
        block_output = coop.block.sum(
            input[cuda.threadIdx.x],
            items_per_thread=1,
        )

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # example-end sum

    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_per_block


def test_block_reduction_temp_storage():
    @cuda.jit(device=True)
    def op(a, b):
        return a + b

    threads_per_block = 128
    block_reduce = coop.block.reduce(
        np.int32,
        threads_per_block,
        op,
        items_per_thread=1,
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes
    temp_storage_alignment = block_reduce.temp_storage_alignment

    @cuda.jit
    def kernel(input, output):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        block_output = coop.block.reduce(
            input[cuda.threadIdx.x],
            items_per_thread=1,
            binary_op=op,
            temp_storage=temp_storage,
        )

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    h_input = np.random.randint(0, 42, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == np.sum(h_input)


def test_block_sum_temp_storage():
    threads_per_block = 128
    block_sum = coop.block.sum(
        np.int32,
        threads_per_block,
        items_per_thread=1,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes
    temp_storage_alignment = block_sum.temp_storage_alignment

    @cuda.jit
    def kernel(input, output):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        block_output = coop.block.sum(
            input[cuda.threadIdx.x],
            items_per_thread=1,
            temp_storage=temp_storage,
        )

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_per_block

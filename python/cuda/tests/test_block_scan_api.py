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


def test_block_exclusive_sum():
    # example-begin exclusive-sum
    items_per_thread = 4
    threads_per_block = 128

    # Specialize exclusive sum for a 1D block of 128 threads owning 4 integer items each
    block_exclusive_sum = cudax.block.exclusive_sum(numba.int32, threads_per_block, items_per_thread)
    temp_storage_bytes = block_exclusive_sum.temp_storage_bytes

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit(link=block_exclusive_sum.files)
    def kernel(data):
        # Allocate shared memory for exclusive sum
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)

        # Obtain a segment of consecutive items that are blocked across threads
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = data[cuda.threadIdx.x * items_per_thread + i]

        # Collectively compute the block-wide exclusive prefix sum
        block_exclusive_sum(temp_storage, thread_data, thread_data)

        # Copy the scanned keys back to the output
        for i in range(items_per_thread):
            data[cuda.threadIdx.x * items_per_thread + i] = thread_data[i]
    # example-end exclusive-sum

    tile_size = threads_per_block * items_per_thread

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == i

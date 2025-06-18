# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda
from pynvjitlink import patch

import cuda.cccl.cooperative.experimental as cudax

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-begin imports
patch.patch_numba_linker(lto=True)
# example-end imports


def test_warp_merge_sort():
    # example-begin merge-sort
    # Define comparison operator
    def compare_op(a, b):
        return a > b

    # Specialize merge sort for a warp of threads owning 4 integer items each
    items_per_thread = 4
    warp_merge_sort = cudax.warp.merge_sort_keys(
        numba.int32, items_per_thread, compare_op
    )

    # Link the merge sort to a CUDA kernel
    @cuda.jit(link=warp_merge_sort.files)
    def kernel(keys):
        # Obtain a segment of consecutive items that are blocked across threads
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)

        for i in range(items_per_thread):
            thread_keys[i] = keys[cuda.threadIdx.x * items_per_thread + i]

        # Collectively sort the keys
        warp_merge_sort(thread_keys)

        # Copy the sorted keys back to the output
        for i in range(items_per_thread):
            keys[cuda.threadIdx.x * items_per_thread + i] = thread_keys[i]

    # example-end merge-sort

    tile_size = 32 * items_per_thread

    h_keys = np.arange(0, tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, 32](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == tile_size - 1 - i

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop

# example-begin imports
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
# example-end imports


def test_block_radix_sort_pairs():
    # example-begin radix-sort-pairs
    items_per_thread = 4
    threads_per_block = 128

    @cuda.jit
    def kernel(keys, values):
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        thread_vals = cuda.local.array(shape=items_per_thread, dtype=numba.int32)

        base = cuda.threadIdx.x * items_per_thread
        for i in range(items_per_thread):
            thread_keys[i] = keys[base + i]
            thread_vals[i] = values[base + i]

        coop.block.radix_sort_pairs(thread_keys, thread_vals, items_per_thread)

        for i in range(items_per_thread):
            keys[base + i] = thread_keys[i]
            values[base + i] = thread_vals[i]

    # example-end radix-sort-pairs

    tile_size = threads_per_block * items_per_thread
    h_keys = np.arange(tile_size - 1, -1, -1, dtype=np.int32)
    h_vals = np.arange(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    d_vals = cuda.to_device(h_vals)

    kernel[1, threads_per_block](d_keys, d_vals)
    h_keys = d_keys.copy_to_host()
    h_vals = d_vals.copy_to_host()

    assert np.all(h_keys[:-1] <= h_keys[1:])
    assert np.all(h_vals == np.arange(tile_size - 1, -1, -1, dtype=np.int32))


def test_block_radix_sort_pairs_descending():
    # example-begin radix-sort-pairs-descending
    items_per_thread = 4
    threads_per_block = 128

    @cuda.jit
    def kernel(keys, values):
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        thread_vals = cuda.local.array(shape=items_per_thread, dtype=numba.int32)

        base = cuda.threadIdx.x * items_per_thread
        for i in range(items_per_thread):
            thread_keys[i] = keys[base + i]
            thread_vals[i] = values[base + i]

        coop.block.radix_sort_pairs_descending(
            thread_keys, thread_vals, items_per_thread
        )

        for i in range(items_per_thread):
            keys[base + i] = thread_keys[i]
            values[base + i] = thread_vals[i]

    # example-end radix-sort-pairs-descending

    tile_size = threads_per_block * items_per_thread
    h_keys = np.arange(tile_size, dtype=np.int32)
    h_vals = np.arange(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    d_vals = cuda.to_device(h_vals)

    kernel[1, threads_per_block](d_keys, d_vals)
    h_keys = d_keys.copy_to_host()
    h_vals = d_vals.copy_to_host()

    assert np.all(h_keys[:-1] >= h_keys[1:])
    assert np.all(h_vals == np.arange(tile_size, dtype=np.int32))

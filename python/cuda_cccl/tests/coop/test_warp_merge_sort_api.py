# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop
from cuda.coop import WarpLoadAlgorithm, WarpStoreAlgorithm

# example-begin imports
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
# example-end imports


def test_warp_merge_sort():
    # example-begin merge-sort
    # Define comparison operator
    @cuda.jit(device=True)
    def compare_op(a, b):
        return a > b

    # Specialize merge sort for a warp of threads owning 4 integer items each
    items_per_thread = 4
    threads_in_warp = 32
    warp_merge_sort = coop.warp.merge_sort_keys(
        numba.int32, items_per_thread, compare_op
    )
    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    # Link the merge sort to a CUDA kernel
    @cuda.jit
    def kernel_two_phase(keys_in, keys_out):
        # Obtain a segment of consecutive items that are blocked across threads
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        warp_load(keys_in, thread_keys)

        # Collectively sort the keys
        warp_merge_sort(thread_keys)

        # Copy the sorted keys back to the output
        warp_store(keys_out, thread_keys)

    # example-end merge-sort

    @cuda.jit
    def kernel_single_phase(keys_in, keys_out):
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        coop.warp.load(
            keys_in,
            thread_keys,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        coop.warp.merge_sort_keys(
            thread_keys,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
        )
        coop.warp.store(
            keys_out,
            thread_keys,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    tile_size = threads_in_warp * items_per_thread

    h_keys = np.arange(0, tile_size, dtype=np.int32)
    d_keys_in = cuda.to_device(h_keys)
    d_keys_two_phase = cuda.device_array_like(d_keys_in)
    d_keys_single_phase = cuda.device_array_like(d_keys_in)
    kernel_two_phase[1, threads_in_warp](d_keys_in, d_keys_two_phase)
    kernel_single_phase[1, threads_in_warp](d_keys_in, d_keys_single_phase)
    h_keys_two_phase = d_keys_two_phase.copy_to_host()
    h_keys_single_phase = d_keys_single_phase.copy_to_host()
    expected = np.sort(h_keys)[::-1]
    for i in range(tile_size):
        assert h_keys_two_phase[i] == expected[i]
        assert h_keys_single_phase[i] == expected[i]
        assert h_keys_two_phase[i] == h_keys_single_phase[i]


def test_warp_merge_sort_thread_data_infers_items_per_thread():
    @cuda.jit(device=True)
    def compare_op(a, b):
        return a > b

    items_per_thread = 4
    threads_in_warp = 32

    @cuda.jit
    def kernel(keys_in, keys_out):
        thread_keys = coop.ThreadData(items_per_thread, dtype=keys_in.dtype)
        coop.warp.load(
            keys_in,
            thread_keys,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        coop.warp.merge_sort_keys(
            thread_keys,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
        )
        coop.warp.store(
            keys_out,
            thread_keys,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    tile_size = threads_in_warp * items_per_thread
    h_keys = np.arange(0, tile_size, dtype=np.int32)
    d_keys_in = cuda.to_device(h_keys)
    d_keys_out = cuda.device_array_like(d_keys_in)
    kernel[1, threads_in_warp](d_keys_in, d_keys_out)
    h_keys_out = d_keys_out.copy_to_host()
    expected = np.sort(h_keys)[::-1]
    np.testing.assert_array_equal(h_keys_out, expected)

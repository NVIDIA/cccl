# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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


def test_warp_merge_sort_pairs():
    # example-begin merge-sort-pairs
    @cuda.jit(device=True)
    def compare_op(a, b):
        return a > b

    items_per_thread = 4
    threads_in_warp = 32
    warp_merge_sort = coop.warp.merge_sort_pairs(
        numba.int32,
        numba.int32,
        items_per_thread,
        compare_op,
    )
    warp_load_keys = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store_keys = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_load_vals = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store_vals = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    @cuda.jit
    def kernel_two_phase(keys_in, values_in, keys_out, values_out):
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        thread_vals = cuda.local.array(shape=items_per_thread, dtype=numba.int32)

        warp_load_keys(keys_in, thread_keys)
        warp_load_vals(values_in, thread_vals)

        warp_merge_sort(thread_keys, thread_vals)

        warp_store_keys(keys_out, thread_keys)
        warp_store_vals(values_out, thread_vals)

    # example-end merge-sort-pairs

    @cuda.jit
    def kernel_single_phase(keys_in, values_in, keys_out, values_out):
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        thread_vals = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        coop.warp.load(
            keys_in,
            thread_keys,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        coop.warp.load(
            values_in,
            thread_vals,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        coop.warp.merge_sort_pairs(
            thread_keys,
            thread_vals,
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
        coop.warp.store(
            values_out,
            thread_vals,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    tile_size = threads_in_warp * items_per_thread
    h_keys = np.arange(tile_size - 1, -1, -1, dtype=np.int32)
    h_vals = np.arange(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    d_vals = cuda.to_device(h_vals)
    d_keys_two_phase = cuda.device_array_like(d_keys)
    d_vals_two_phase = cuda.device_array_like(d_vals)
    d_keys_single_phase = cuda.device_array_like(d_keys)
    d_vals_single_phase = cuda.device_array_like(d_vals)

    kernel_two_phase[1, threads_in_warp](
        d_keys, d_vals, d_keys_two_phase, d_vals_two_phase
    )
    kernel_single_phase[1, threads_in_warp](
        d_keys, d_vals, d_keys_single_phase, d_vals_single_phase
    )
    h_keys_two_phase = d_keys_two_phase.copy_to_host()
    h_vals_two_phase = d_vals_two_phase.copy_to_host()
    h_keys_single_phase = d_keys_single_phase.copy_to_host()
    h_vals_single_phase = d_vals_single_phase.copy_to_host()

    expected_order = np.argsort(-h_keys, kind="stable")
    expected_keys = h_keys[expected_order]
    expected_vals = h_vals[expected_order]

    assert np.all(h_keys_two_phase[:-1] >= h_keys_two_phase[1:])
    assert np.all(h_keys_single_phase[:-1] >= h_keys_single_phase[1:])
    np.testing.assert_array_equal(h_keys_two_phase, expected_keys)
    np.testing.assert_array_equal(h_keys_single_phase, expected_keys)
    np.testing.assert_array_equal(h_vals_two_phase, expected_vals)
    np.testing.assert_array_equal(h_vals_single_phase, expected_vals)

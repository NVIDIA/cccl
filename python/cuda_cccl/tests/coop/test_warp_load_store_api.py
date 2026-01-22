# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_warp_load_store():
    threads_in_warp = 32
    items_per_thread = 4

    warp_load = coop.warp.load.create(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    warp_store = coop.warp.store.create(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )

    @cuda.jit(link=warp_load.files + warp_store.files)
    def kernel(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        warp_store(d_out, items)

    h_input = np.random.randint(
        0, 42, threads_in_warp * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_in_warp](d_input, d_output)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_warp_load_store_num_valid_oob_default():
    threads_in_warp = 32
    items_per_thread = 2
    total_items = threads_in_warp * items_per_thread
    num_valid = total_items - 7
    oob_default = np.int32(-123)
    sentinel = np.int32(-999)

    warp_load = coop.warp.load.create(
        numba.int32,
        items_per_thread,
        threads_in_warp,
        algorithm="striped",
        num_valid_items=num_valid,
        oob_default=oob_default,
    )
    warp_store_all = coop.warp.store.create(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    warp_store_valid = coop.warp.store.create(
        numba.int32,
        items_per_thread,
        threads_in_warp,
        algorithm="striped",
        num_valid_items=num_valid,
    )

    @cuda.jit(link=warp_load.files + warp_store_all.files)
    def kernel_all(d_in, d_out_all):
        items = cuda.local.array(items_per_thread, numba.int32)
        num_valid_items = numba.int32(num_valid)
        oob = numba.int32(oob_default)
        warp_load(d_in, items, num_valid_items, oob)
        warp_store_all(d_out_all, items)

    @cuda.jit(link=warp_load.files + warp_store_valid.files)
    def kernel_valid(d_in, d_out_valid):
        items = cuda.local.array(items_per_thread, numba.int32)
        num_valid_items = numba.int32(num_valid)
        oob = numba.int32(oob_default)
        warp_load(d_in, items, num_valid_items, oob)
        warp_store_valid(d_out_valid, items, num_valid_items)

    h_input = np.arange(total_items, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_all = cuda.device_array_like(d_input)
    d_out_valid = cuda.to_device(np.full(total_items, sentinel, dtype=np.int32))

    kernel_all[1, threads_in_warp](d_input, d_out_all)
    kernel_valid[1, threads_in_warp](d_input, d_out_valid)
    h_out_all = d_out_all.copy_to_host()
    h_out_valid = d_out_valid.copy_to_host()

    expected_all = np.full(total_items, oob_default, dtype=np.int32)
    expected_all[:num_valid] = h_input[:num_valid]

    expected_valid = np.full(total_items, sentinel, dtype=np.int32)
    expected_valid[:num_valid] = h_input[:num_valid]

    np.testing.assert_array_equal(h_out_all, expected_all)
    np.testing.assert_array_equal(h_out_valid, expected_valid)

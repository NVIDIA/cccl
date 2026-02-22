# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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


def test_warp_load_store():
    # example-begin load-store
    threads_in_warp = 32
    items_per_thread = 4

    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )

    @cuda.jit
    def kernel(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        warp_store(d_out, items)

    # example-end load-store

    h_input = np.random.randint(
        0, 42, threads_in_warp * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_in_warp](d_input, d_output)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_warp_load_store_temp_storage():
    threads_in_warp = 32
    items_per_thread = 4

    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    temp_storage_bytes = max(
        warp_load.temp_storage_bytes,
        warp_store.temp_storage_bytes,
    )
    temp_storage_alignment = max(
        warp_load.temp_storage_alignment,
        warp_store.temp_storage_alignment,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items, temp_storage=temp_storage)
        warp_store(d_out, items, temp_storage=temp_storage)

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

    warp_load = coop.warp.load(
        numba.int32,
        items_per_thread,
        threads_in_warp,
        algorithm="striped",
        num_valid_items=num_valid,
        oob_default=oob_default,
    )
    warp_store_all = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    warp_store_valid = coop.warp.store(
        numba.int32,
        items_per_thread,
        threads_in_warp,
        algorithm="striped",
        num_valid_items=num_valid,
    )

    # example-begin load-store-num-valid-oob-default
    @cuda.jit
    def kernel_all(d_in, d_out_all):
        items = cuda.local.array(items_per_thread, numba.int32)
        num_valid_items = numba.int32(num_valid)
        oob = numba.int32(oob_default)
        warp_load(d_in, items, num_valid_items, oob)
        warp_store_all(d_out_all, items)

    @cuda.jit
    def kernel_valid(d_in, d_out_valid):
        items = cuda.local.array(items_per_thread, numba.int32)
        num_valid_items = numba.int32(num_valid)
        oob = numba.int32(oob_default)
        warp_load(d_in, items, num_valid_items, oob)
        warp_store_valid(d_out_valid, items, num_valid_items)

    # example-end load-store-num-valid-oob-default

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


def test_warp_load_store_single_phase():
    threads_in_warp = 32
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.STRIPED,
        )
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.STRIPED,
        )

    h_input = np.random.randint(
        0, 42, threads_in_warp * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_warp_load_store_single_phase_thread_data_infers_items_per_thread():
    threads_in_warp = 32
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        coop.warp.load(
            d_in,
            items,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.STRIPED,
        )
        coop.warp.store(
            d_out,
            items,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.STRIPED,
        )

    h_input = np.random.randint(
        0, 42, threads_in_warp * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_warp_load_store_temp_storage_single_phase():
    threads_in_warp = 32
    items_per_thread = 4

    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="striped"
    )
    temp_storage_bytes = max(
        warp_load.temp_storage_bytes,
        warp_store.temp_storage_bytes,
    )
    temp_storage_alignment = max(
        warp_load.temp_storage_alignment,
        warp_store.temp_storage_alignment,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = cuda.local.array(items_per_thread, numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.STRIPED,
            temp_storage=temp_storage,
        )
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.STRIPED,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_in_warp * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_warp_load_store_num_valid_oob_default_single_phase():
    threads_in_warp = 32
    items_per_thread = 2
    total_items = threads_in_warp * items_per_thread
    num_valid = total_items - 7
    oob_default = np.int32(-123)
    sentinel = np.int32(-999)

    @cuda.jit
    def kernel_all(d_in, d_out_all):
        items = cuda.local.array(items_per_thread, numba.int32)
        num_valid_items = numba.int32(num_valid)
        oob = numba.int32(oob_default)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.STRIPED,
            num_valid_items=num_valid_items,
            oob_default=oob,
        )
        coop.warp.store(
            d_out_all,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.STRIPED,
        )

    @cuda.jit
    def kernel_valid(d_in, d_out_valid):
        items = cuda.local.array(items_per_thread, numba.int32)
        num_valid_items = numba.int32(num_valid)
        oob = numba.int32(oob_default)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.STRIPED,
            num_valid_items=num_valid_items,
            oob_default=oob,
        )
        coop.warp.store(
            d_out_valid,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.STRIPED,
            num_valid_items=num_valid_items,
        )

    h_input = np.arange(total_items, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_all = cuda.device_array_like(d_input)
    d_out_valid = cuda.to_device(np.full(total_items, sentinel, dtype=np.int32))

    kernel_all[1, threads_in_warp](d_input, d_out_all)
    kernel_valid[1, threads_in_warp](d_input, d_out_valid)
    cuda.synchronize()
    h_out_all = d_out_all.copy_to_host()
    h_out_valid = d_out_valid.copy_to_host()

    expected_all = np.full(total_items, oob_default, dtype=np.int32)
    expected_all[:num_valid] = h_input[:num_valid]

    expected_valid = np.full(total_items, sentinel, dtype=np.int32)
    expected_valid[:num_valid] = h_input[:num_valid]

    np.testing.assert_array_equal(h_out_all, expected_all)
    np.testing.assert_array_equal(h_out_valid, expected_valid)

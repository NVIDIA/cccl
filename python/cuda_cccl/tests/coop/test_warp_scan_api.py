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


def test_warp_exclusive_sum():
    # example-begin exclusive-sum
    # Specialize exclusive sum for a warp of threads
    warp_exclusive_sum = coop.warp.exclusive_sum(numba.int32)
    threads_in_warp = 32
    items_per_thread = 1
    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit
    def kernel_two_phase(d_in, d_out):
        # Collectively compute the warp-wide exclusive prefix sum
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        items[0] = warp_exclusive_sum(items[0])
        warp_store(d_out, items)

    # example-end exclusive-sum

    @cuda.jit
    def kernel_single_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        items[0] = coop.warp.exclusive_sum(items[0])
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    tile_size = threads_in_warp

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_input = cuda.to_device(h_keys)
    d_out_two_phase = cuda.device_array_like(d_input)
    d_out_single_phase = cuda.device_array_like(d_input)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)
    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()
    for i in range(tile_size):
        assert h_out_two_phase[i] == i
        assert h_out_single_phase[i] == i
        assert h_out_two_phase[i] == h_out_single_phase[i]


test_warp_exclusive_sum()


def test_warp_inclusive_sum():
    # example-begin inclusive-sum
    warp_inclusive_sum = coop.warp.inclusive_sum(numba.int32)
    threads_in_warp = 32
    items_per_thread = 1
    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    @cuda.jit
    def kernel_two_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        items[0] = warp_inclusive_sum(items[0])
        warp_store(d_out, items)

    # example-end inclusive-sum

    @cuda.jit
    def kernel_single_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        items[0] = coop.warp.inclusive_sum(items[0])
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    tile_size = threads_in_warp
    h_keys = np.ones(tile_size, dtype=np.int32)
    d_input = cuda.to_device(h_keys)
    d_out_two_phase = cuda.device_array_like(d_input)
    d_out_single_phase = cuda.device_array_like(d_input)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)
    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()
    for i in range(tile_size):
        assert h_out_two_phase[i] == i + 1
        assert h_out_single_phase[i] == i + 1
        assert h_out_two_phase[i] == h_out_single_phase[i]


test_warp_inclusive_sum()


def test_warp_exclusive_scan():
    # example-begin exclusive-scan
    @cuda.jit(device=True)
    def op(a, b):
        return a + b

    warp_scan = coop.warp.exclusive_scan(numba.int32, op)
    threads_in_warp = 32
    items_per_thread = 1
    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    @cuda.jit
    def kernel_two_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        items[0] = warp_scan(items[0])
        warp_store(d_out, items)

    # example-end exclusive-scan

    @cuda.jit
    def kernel_single_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        items[0] = coop.warp.exclusive_scan(items[0], op)
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    tile_size = threads_in_warp
    h_keys = np.ones(tile_size, dtype=np.int32)
    d_input = cuda.to_device(h_keys)
    d_out_two_phase = cuda.device_array_like(d_input)
    d_out_single_phase = cuda.device_array_like(d_input)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)
    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()
    for i in range(tile_size):
        assert h_out_two_phase[i] == i
        assert h_out_single_phase[i] == i
        assert h_out_two_phase[i] == h_out_single_phase[i]


def test_warp_inclusive_scan():
    # example-begin inclusive-scan
    @cuda.jit(device=True)
    def op(a, b):
        return a + b

    warp_scan = coop.warp.inclusive_scan(numba.int32, op)
    threads_in_warp = 32
    items_per_thread = 1
    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    @cuda.jit
    def kernel_two_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        items[0] = warp_scan(items[0])
        warp_store(d_out, items)

    # example-end inclusive-scan

    @cuda.jit
    def kernel_single_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.DIRECT,
        )
        items[0] = coop.warp.inclusive_scan(items[0], op)
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    tile_size = threads_in_warp
    h_keys = np.ones(tile_size, dtype=np.int32)
    d_input = cuda.to_device(h_keys)
    d_out_two_phase = cuda.device_array_like(d_input)
    d_out_single_phase = cuda.device_array_like(d_input)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)
    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()
    for i in range(tile_size):
        assert h_out_two_phase[i] == i + 1
        assert h_out_single_phase[i] == i + 1
        assert h_out_two_phase[i] == h_out_single_phase[i]

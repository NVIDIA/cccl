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


def test_warp_reduction():
    @cuda.jit(device=True)
    def op(a, b):
        return a if a > b else b

    # example-begin reduce
    warp_reduce = coop.warp.reduce(numba.int32, op)
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
        items[0] = warp_reduce(items[0])
        warp_store(d_out, items)

    # example-end reduce

    h_input = np.random.randint(0, 42, 32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_two_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    d_out_single_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)

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
        items[0] = coop.warp.reduce(items[0], op)
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()
    h_expected = np.max(h_input)

    assert h_out_two_phase[0] == h_expected
    assert h_out_single_phase[0] == h_expected
    assert h_out_two_phase[0] == h_out_single_phase[0]


def test_warp_sum():
    # example-begin sum
    warp_sum = coop.warp.sum(numba.int32)
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
        items[0] = warp_sum(items[0])
        warp_store(d_out, items)

    # example-end sum

    h_input = np.ones(32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_two_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    d_out_single_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)

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
        items[0] = coop.warp.sum(items[0])
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()

    assert h_out_two_phase[0] == 32
    assert h_out_single_phase[0] == 32
    assert h_out_two_phase[0] == h_out_single_phase[0]


def test_warp_sum_valid_items():
    warp_sum = coop.warp.sum(numba.int32)
    valid_items = 8
    threads_in_warp = 32
    items_per_thread = 1
    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    # example-begin sum-valid-items
    @cuda.jit
    def kernel_two_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        items[0] = warp_sum(items[0], valid_items=valid_items)
        warp_store(d_out, items)

    # example-end sum-valid-items

    h_input = np.arange(32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_two_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    d_out_single_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)

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
        items[0] = coop.warp.sum(items[0], valid_items=valid_items)
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()

    expected = np.sum(h_input[:valid_items])
    assert h_out_two_phase[0] == expected
    assert h_out_single_phase[0] == expected
    assert h_out_two_phase[0] == h_out_single_phase[0]


def test_warp_reduce_valid_items():
    @cuda.jit(device=True)
    def op(a, b):
        return a if a > b else b

    warp_reduce = coop.warp.reduce(numba.int32, op)
    valid_items = 12
    threads_in_warp = 32
    items_per_thread = 1
    warp_load = coop.warp.load(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )
    warp_store = coop.warp.store(
        numba.int32, items_per_thread, threads_in_warp, algorithm="direct"
    )

    # example-begin reduce-valid-items
    @cuda.jit
    def kernel_two_phase(d_in, d_out):
        items = cuda.local.array(items_per_thread, numba.int32)
        warp_load(d_in, items)
        items[0] = warp_reduce(items[0], valid_items=valid_items)
        warp_store(d_out, items)

    # example-end reduce-valid-items

    h_input = np.random.randint(0, 42, 32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_two_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    d_out_single_phase = cuda.device_array(threads_in_warp, dtype=np.int32)
    kernel_two_phase[1, threads_in_warp](d_input, d_out_two_phase)

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
        items[0] = coop.warp.reduce(items[0], op, valid_items=valid_items)
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.DIRECT,
        )

    kernel_single_phase[1, threads_in_warp](d_input, d_out_single_phase)
    h_out_two_phase = d_out_two_phase.copy_to_host()
    h_out_single_phase = d_out_single_phase.copy_to_host()

    expected = np.max(h_input[:valid_items])
    assert h_out_two_phase[0] == expected
    assert h_out_single_phase[0] == expected
    assert h_out_two_phase[0] == h_out_single_phase[0]

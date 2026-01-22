# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest
from helpers import NUMBA_TYPES_TO_NP, random_int
from numba import cuda, types

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_warp_reduce_sum_single_phase():
    @cuda.jit(device=True)
    def max_op(a, b):
        return a if a > b else b

    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out_max, d_out_sum):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_max = coop.warp.reduce(val, max_op)
        warp_sum = coop.warp.sum(val)
        if tid == 0:
            d_out_max[0] = warp_max
            d_out_sum[0] = warp_sum

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_max = cuda.device_array(1, dtype=np.int32)
    d_out_sum = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_out_max, d_out_sum)
    cuda.synchronize()

    h_out_max = d_out_max.copy_to_host()
    h_out_sum = d_out_sum.copy_to_host()

    assert h_out_max[0] == np.max(h_input)
    assert h_out_sum[0] == np.sum(h_input)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_exclusive_sum_single_phase(T):
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        d_out[tid] = coop.warp.exclusive_sum(val)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = np.ones(threads_in_warp, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.arange(threads_in_warp, dtype=dtype)
    np.testing.assert_array_equal(h_output, expected)


def test_warp_load_store_single_phase():
    threads_in_warp = 32
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        items = cuda.local.array(items_per_thread, dtype=numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm="striped",
        )
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm="striped",
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


def test_warp_merge_sort_single_phase():
    T = types.int32
    items_per_thread = 2
    threads_in_warp = 32

    def op(a, b):
        return a < b

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = d_in[tid * items_per_thread + i]
        coop.warp.merge_sort_keys(
            thread_data,
            items_per_thread=items_per_thread,
            compare_op=op,
            threads_in_warp=threads_in_warp,
        )
        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = threads_in_warp * items_per_thread
    h_input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = sorted(h_input)
    for i in range(items_per_tile):
        assert h_output[i] == reference[i]

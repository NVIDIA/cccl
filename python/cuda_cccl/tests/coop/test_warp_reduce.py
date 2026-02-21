# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest
from helpers import NUMBA_TYPES_TO_NP, random_int
from numba import cuda, types

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
def test_warp_reduction_of_integral_type(T):
    def op(a, b):
        return a if a < b else b

    warp_reduce = coop.warp.reduce(T, op)

    @cuda.jit
    def kernel(input, output):
        warp_output = warp_reduce(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
def test_warp_sum(T):
    warp_reduce = coop.warp.sum(T)

    @cuda.jit
    def kernel(input, output):
        warp_output = warp_reduce(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


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


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
def test_warp_reduce_min_single_phase_unsigned(T):
    @cuda.jit(device=True)
    def min_op(a, b):
        return a if a < b else b

    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_min = coop.warp.reduce(val, min_op)
        if tid == 0:
            d_out[0] = warp_min

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_warp, dtype)
    d_input = cuda.to_device(h_input)
    d_out = cuda.device_array(1, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_out)
    cuda.synchronize()

    h_out = d_out.copy_to_host()
    assert h_out[0] == np.min(h_input)


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
def test_warp_sum_single_phase_unsigned(T):
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_sum = coop.warp.sum(val)
        if tid == 0:
            d_out[0] = warp_sum

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_warp, dtype)
    d_input = cuda.to_device(h_input)
    d_out = cuda.device_array(1, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_out)
    cuda.synchronize()

    h_out = d_out.copy_to_host()
    assert h_out[0] == np.sum(h_input)


def test_warp_sum_single_phase_valid_items():
    threads_in_warp = 32
    valid_items = 10

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_sum = coop.warp.sum(val, valid_items=valid_items)
        if tid == 0:
            d_out[0] = warp_sum

    h_input = np.arange(threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    assert h_output[0] == np.sum(h_input[:valid_items])


def test_warp_reduce_single_phase_valid_items():
    @cuda.jit(device=True)
    def max_op(a, b):
        return a if a > b else b

    threads_in_warp = 32
    valid_items = 7

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_max = coop.warp.reduce(val, max_op, valid_items=valid_items)
        if tid == 0:
            d_out[0] = warp_max

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    assert h_output[0] == np.max(h_input[:valid_items])


def test_warp_reduce_temp_storage_single_phase():
    @cuda.jit(device=True)
    def max_op_single(a, b):
        return a if a > b else b

    @cuda.jit(device=True)
    def max_op_two_phase(a, b):
        return a if a > b else b

    threads_in_warp = 32
    warp_reduce = coop.warp.reduce(types.int32, max_op_two_phase)
    temp_storage_bytes = warp_reduce.temp_storage_bytes
    temp_storage_alignment = warp_reduce.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out_single, d_out_two_phase):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        warp_out_single = coop.warp.reduce(
            val, max_op_single, temp_storage=temp_storage
        )
        warp_out_two_phase = warp_reduce(val, temp_storage=temp_storage)
        if tid == 0:
            d_out_single[0] = warp_out_single
            d_out_two_phase[0] = warp_out_two_phase

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_single = cuda.device_array(1, dtype=np.int32)
    d_out_two_phase = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_out_single, d_out_two_phase)
    cuda.synchronize()

    h_out_single = d_out_single.copy_to_host()
    h_out_two_phase = d_out_two_phase.copy_to_host()
    expected = np.max(h_input)
    assert h_out_single[0] == expected
    assert h_out_two_phase[0] == expected


def test_warp_sum_temp_storage_single_phase():
    threads_in_warp = 32
    warp_sum = coop.warp.sum(types.int32)
    temp_storage_bytes = warp_sum.temp_storage_bytes
    temp_storage_alignment = warp_sum.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out_single, d_out_two_phase):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        warp_out_single = coop.warp.sum(val, temp_storage=temp_storage)
        warp_out_two_phase = warp_sum(val, temp_storage=temp_storage)
        if tid == 0:
            d_out_single[0] = warp_out_single
            d_out_two_phase[0] = warp_out_two_phase

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_single = cuda.device_array(1, dtype=np.int32)
    d_out_two_phase = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_out_single, d_out_two_phase)
    cuda.synchronize()

    h_out_single = d_out_single.copy_to_host()
    h_out_two_phase = d_out_two_phase.copy_to_host()
    expected = np.sum(h_input)
    assert h_out_single[0] == expected
    assert h_out_two_phase[0] == expected


def test_warp_sum_temp_storage_getitem_sugar():
    threads_in_warp = 32

    @cuda.jit
    def kernel(input, output):
        val = input[cuda.threadIdx.x]
        temp_storage = coop.TempStorage()
        warp_out = coop.warp.sum[temp_storage](val)
        if cuda.threadIdx.x == 0:
            output[0] = warp_out

    h_input = np.random.randint(0, 42, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    assert h_output[0] == np.sum(h_input)

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pynvjitlink import patch
import cuda.cooperative.experimental as cudax
from helpers import random_int, NUMBA_TYPES_TO_NP
import pytest
from numba import cuda, types
import numba
patch.patch_numba_linker(lto=True)
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize('T', [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize('threads_in_block', [32, 128, 256, 1024])
@pytest.mark.parametrize('items_per_thread', [1, 3])
def test_block_radix_sort_descending(T, threads_in_block, items_per_thread):
    begin_bit = numba.int32(0)
    end_bit = numba.int32(T.bitwidth)
    block_radix_sort = cudax.block.radix_sort_keys_descending(
        dtype=T, threads_in_block=threads_in_block, items_per_thread=items_per_thread)
    temp_storage_bytes = block_radix_sort.temp_storage_bytes

    @cuda.jit(link=block_radix_sort.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_radix_sort(temp_storage, thread_data, begin_bit, end_bit)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = threads_in_block * items_per_thread
    input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input, reverse=True)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass



@pytest.mark.parametrize('T', [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize('threads_in_block', [32, 128, 256, 1024])
@pytest.mark.parametrize('items_per_thread', [1, 3])
def test_block_radix_sort(T, threads_in_block, items_per_thread):
    items_per_tile = threads_in_block * items_per_thread

    block_radix_sort = cudax.block.radix_sort_keys(
        dtype=T, threads_in_block=threads_in_block, items_per_thread=items_per_thread)
    temp_storage_bytes = block_radix_sort.temp_storage_bytes

    @cuda.jit(link=block_radix_sort.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_radix_sort(temp_storage, thread_data)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass


def test_block_radix_sort_overloads_work():
    T = numba.int32
    threads_in_block = 128
    items_per_thread = 3
    items_per_tile = threads_in_block * items_per_thread

    block_radix_sort = cudax.block.radix_sort_keys(
        dtype=T, threads_in_block=threads_in_block, items_per_thread=items_per_thread)
    temp_storage_bytes = block_radix_sort.temp_storage_bytes

    @cuda.jit(link=block_radix_sort.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype='int32')
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_radix_sort(temp_storage, thread_data, numba.int32(0), numba.int32(32))
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input)
    for i in range(items_per_tile):
        assert output[i] == reference[i]


def test_block_radix_sort_mangling():
    return # TODO Return to linker issue
    threads_in_block = 128
    items_per_thread = 3
    items_per_tile = threads_in_block * items_per_thread

    int_block_radix_sort = cudax.block.radix_sort_keys(
        dtype=numba.int32, threads_in_block=threads_in_block, items_per_thread=items_per_thread)
    int_temp_storage_bytes = int_block_radix_sort.temp_storage_bytes

    double_block_radix_sort = cudax.block.radix_sort_keys(
        dtype=numba.float64, threads_in_block=threads_in_block, items_per_thread=items_per_thread)
    double_temp_storage_bytes = double_block_radix_sort.temp_storage_bytes

    @cuda.jit(link=int_block_radix_sort.files + double_block_radix_sort.files)
    def kernel(int_input, int_output, double_input, double_output):
        tid = cuda.threadIdx.x
        int_temp_storage = cuda.shared.array(shape=int_temp_storage_bytes, dtype='uint8')
        int_thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            int_thread_data[i] = int_input[tid * items_per_thread + i]
        int_block_radix_sort(int_temp_storage, int_thread_data)
        for i in range(items_per_thread):
            int_output[tid * items_per_thread + i] = int_thread_data[i]
        double_temp_storage = cuda.shared.array(shape=double_temp_storage_bytes, dtype='uint8')
        double_thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.float64)
        for i in range(items_per_thread):
            double_thread_data[i] = double_input[tid * items_per_thread + i]
        double_block_radix_sort(double_temp_storage, double_thread_data)
        for i in range(items_per_thread):
            double_output[tid * items_per_thread + i] = double_thread_data[i]

    int_input = random_int(items_per_tile, 'int32')
    d_int_input = cuda.to_device(int_input)
    d_int_output = cuda.device_array(items_per_tile, dtype='int32')
    double_input = random_int(items_per_tile, 'float64')
    d_double_input = cuda.to_device(double_input)
    d_double_output = cuda.device_array(items_per_tile, dtype='float64')
    kernel[1, threads_in_block](d_int_input, d_int_output, d_double_input, d_double_output)
    cuda.synchronize()

    int_output = d_int_output.copy_to_host()
    int_reference = sorted(int_input)
    for i in range(items_per_tile):
        assert int_output[i] == int_reference[i]

    double_output = d_double_output.copy_to_host()
    double_reference = sorted(double_input)
    for i in range(items_per_tile):
        assert double_output[i] == double_reference[i]

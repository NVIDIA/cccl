# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pynvjitlink import patch
import numpy as np
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
def test_block_merge_sort(T, threads_in_block, items_per_thread):
    def op(a, b):
        return a < b

    block_merge_sort = cudax.block.merge_sort_keys(
        dtype=T, threads_in_block=threads_in_block, items_per_thread=items_per_thread, compare_op=op)
    temp_storage_bytes = block_merge_sort.temp_storage_bytes

    @cuda.jit(link=block_merge_sort.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(
            shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_merge_sort(temp_storage, thread_data)
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
    reference = sorted(input)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass


@pytest.mark.parametrize('T', [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize('threads_in_block', [32, 128, 256, 1024])
@pytest.mark.parametrize('items_per_thread', [1, 3])
def test_block_merge_sort_descending(T, threads_in_block, items_per_thread):
    def op(a, b):
        return a > b

    block_merge_sort = cudax.block.merge_sort_keys(
        dtype=T, threads_in_block=threads_in_block, items_per_thread=items_per_thread, compare_op=op)
    temp_storage_bytes = block_merge_sort.temp_storage_bytes

    @cuda.jit(link=block_merge_sort.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(
            shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_merge_sort(temp_storage, thread_data)
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


def test_block_merge_sort_user_defined_type():
    items_per_thread = 3
    threads_in_block = 128
    items_per_tile = threads_in_block * items_per_thread

    def op(a, b):
        return a[0].real > b[0].real

    block_merge_sort = cudax.block.merge_sort_keys(
        dtype=numba.complex128, threads_in_block=threads_in_block, items_per_thread=items_per_thread, compare_op=op)
    temp_storage_bytes = block_merge_sort.temp_storage_bytes

    @cuda.jit(link=block_merge_sort.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(
            shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_merge_sort(temp_storage, thread_data)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = np.complex128
    items_per_tile = threads_in_block * items_per_thread
    input = np.random.random(items_per_tile) + 1j * \
        np.random.random(items_per_tile)
    input = input.astype(dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input, reverse=True, key=lambda x: x.real)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

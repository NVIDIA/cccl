# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
import pytest
from helpers import NUMBA_TYPES_TO_NP, random_int, row_major_tid
from numba import cuda, types

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_merge_sort_two_phase():
    @cuda.jit(device=True)
    def op(a, b):
        return a > b

    threads_per_block = 64
    items_per_thread = 2
    dtype = np.int32

    block_merge_sort = coop.block.merge_sort_keys(
        numba.int32,
        threads_per_block,
        items_per_thread,
        op,
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_merge_sort(thread_data, items_per_thread, op)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    items_per_tile = threads_per_block * items_per_thread
    h_input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = sorted(h_input, reverse=True)
    np.testing.assert_array_equal(h_output, reference)


def test_block_merge_sort_temp_storage():
    @cuda.jit(device=True)
    def op(a, b):
        return a < b

    threads_per_block = 64
    items_per_thread = 2
    dtype = np.int32

    block_merge_sort = coop.block.merge_sort_keys(
        numba.int32,
        threads_per_block,
        items_per_thread,
        op,
    )
    temp_storage_bytes = block_merge_sort.temp_storage_bytes
    temp_storage_alignment = block_merge_sort.temp_storage_alignment

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_merge_sort(thread_data, items_per_thread, op, temp_storage=temp_storage)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    items_per_tile = threads_per_block * items_per_thread
    h_input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = sorted(h_input)
    np.testing.assert_array_equal(h_output, reference)


def test_block_merge_sort_key_value():
    @cuda.jit(device=True)
    def op(a, b):
        return a < b

    threads_per_block = 64
    items_per_thread = 2
    items_per_tile = threads_per_block * items_per_thread
    dtype = np.int32

    @cuda.jit
    def kernel(keys_in, values_in, keys_out, values_out):
        tid = row_major_tid()
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        thread_values = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            thread_keys[i] = keys_in[idx]
            thread_values[i] = values_in[idx]

        coop.block.merge_sort_keys(
            thread_keys,
            items_per_thread,
            op,
            values=thread_values,
        )

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            keys_out[idx] = thread_keys[i]
            values_out[idx] = thread_values[i]

    h_keys = random_int(items_per_tile, dtype)
    h_values = (h_keys * 3 + 7).astype(dtype)
    d_keys = cuda.to_device(h_keys)
    d_values = cuda.to_device(h_values)
    d_keys_out = cuda.device_array(items_per_tile, dtype=dtype)
    d_values_out = cuda.device_array(items_per_tile, dtype=dtype)

    kernel[1, threads_per_block](d_keys, d_values, d_keys_out, d_values_out)
    cuda.synchronize()

    keys_out = d_keys_out.copy_to_host()
    values_out = d_values_out.copy_to_host()

    expected = sorted(zip(h_keys, h_values), key=lambda kv: kv[0])
    exp_keys = np.array([kv[0] for kv in expected], dtype=dtype)
    exp_values = np.array([kv[1] for kv in expected], dtype=dtype)

    np.testing.assert_array_equal(keys_out, exp_keys)
    np.testing.assert_array_equal(values_out, exp_values)


def test_block_merge_sort_key_value_two_phase():
    @cuda.jit(device=True)
    def op(a, b):
        return a < b

    threads_per_block = 64
    items_per_thread = 2
    items_per_tile = threads_per_block * items_per_thread
    dtype = np.int32

    block_merge_sort = coop.block.merge_sort_keys(
        numba.int32, threads_per_block, items_per_thread, op
    )

    @cuda.jit
    def kernel(keys_in, values_in, keys_out, values_out):
        tid = row_major_tid()
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        thread_values = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            thread_keys[i] = keys_in[idx]
            thread_values[i] = values_in[idx]

        block_merge_sort(
            thread_keys,
            items_per_thread,
            op,
            values=thread_values,
        )

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            keys_out[idx] = thread_keys[i]
            values_out[idx] = thread_values[i]

    h_keys = random_int(items_per_tile, dtype)
    h_values = (h_keys * 5 + 1).astype(dtype)
    d_keys = cuda.to_device(h_keys)
    d_values = cuda.to_device(h_values)
    d_keys_out = cuda.device_array(items_per_tile, dtype=dtype)
    d_values_out = cuda.device_array(items_per_tile, dtype=dtype)

    kernel[1, threads_per_block](d_keys, d_values, d_keys_out, d_values_out)
    cuda.synchronize()

    keys_out = d_keys_out.copy_to_host()
    values_out = d_values_out.copy_to_host()

    expected = sorted(zip(h_keys, h_values), key=lambda kv: kv[0])
    exp_keys = np.array([kv[0] for kv in expected], dtype=dtype)
    exp_values = np.array([kv[1] for kv in expected], dtype=dtype)

    np.testing.assert_array_equal(keys_out, exp_keys)
    np.testing.assert_array_equal(values_out, exp_values)


def test_block_merge_sort_valid_items():
    @cuda.jit(device=True)
    def op(a, b):
        return a < b

    threads_per_block = 64
    items_per_thread = 2
    items_per_tile = threads_per_block * items_per_thread
    valid_items = items_per_tile - 3
    dtype = np.int32
    oob_default = numba.int32(9999)

    @cuda.jit
    def kernel(input_keys, output_keys):
        tid = row_major_tid()
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            thread_keys[i] = input_keys[idx]

        coop.block.merge_sort_keys(
            thread_keys,
            items_per_thread,
            op,
            valid_items=valid_items,
            oob_default=oob_default,
        )

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            output_keys[idx] = thread_keys[i]

    h_keys = random_int(items_per_tile, dtype)
    d_keys = cuda.to_device(h_keys)
    d_out = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_keys, d_out)
    cuda.synchronize()

    h_out = d_out.copy_to_host()
    reference = sorted(h_keys[:valid_items])
    np.testing.assert_array_equal(h_out[:valid_items], np.array(reference, dtype=dtype))


@pytest.mark.parametrize("T", [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize("threads_per_block", [32, 128, 256, 1024, (8, 16), (2, 4, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 3])
def test_block_merge_sort(T, threads_per_block, items_per_thread):
    @cuda.jit(device=True)
    def op(a, b):
        return a < b

    num_threads_per_block = (
        threads_per_block
        if type(threads_per_block) is int
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        coop.block.merge_sort_keys(thread_data, items_per_thread, op)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads_per_block * items_per_thread
    input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize("threads_per_block", [32, 128, 256, 1024, (8, 16), (2, 4, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 3])
def test_block_merge_sort_descending(T, threads_per_block, items_per_thread):
    @cuda.jit(device=True)
    def op(a, b):
        return a > b

    num_threads_per_block = (
        threads_per_block
        if type(threads_per_block) is int
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        coop.block.merge_sort_keys(thread_data, items_per_thread, op)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads_per_block * items_per_thread
    input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input, reverse=True)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


def test_block_merge_sort_user_defined_type():
    items_per_thread = 3
    threads_per_block = 128
    items_per_tile = threads_per_block * items_per_thread

    @cuda.jit(device=True)
    def op(a, b):
        return a[0].real > b[0].real

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        coop.block.merge_sort_keys(thread_data, items_per_thread, op)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = np.complex128
    items_per_tile = threads_per_block * items_per_thread
    input = np.random.random(items_per_tile) + 1j * np.random.random(items_per_tile)
    input = input.astype(dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input, reverse=True, key=lambda x: x.real)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

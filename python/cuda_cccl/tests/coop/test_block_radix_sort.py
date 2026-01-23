# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
import pytest
from helpers import (
    NUMBA_TYPES_TO_NP,
    Complex,
    complex_type,
    random_int,
    row_major_tid,
)
from numba import cuda, types

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_radix_sort_two_phase():
    threads_per_block = 64
    items_per_thread = 2
    dtype = np.int32
    begin_bit = numba.int32(0)
    end_bit = numba.int32(32)

    block_radix_sort = coop.block.radix_sort_keys(
        numba.int32,
        threads_per_block,
        items_per_thread,
        begin_bit,
        end_bit,
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_radix_sort(thread_data, items_per_thread, begin_bit, end_bit)
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


def test_block_radix_sort_descending_two_phase():
    threads_per_block = 64
    items_per_thread = 2
    dtype = np.int32
    begin_bit = numba.int32(0)
    end_bit = numba.int32(32)

    block_radix_sort_desc = coop.block.radix_sort_keys_descending(
        numba.int32,
        threads_per_block,
        items_per_thread,
        begin_bit,
        end_bit,
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_radix_sort_desc(thread_data, items_per_thread, begin_bit, end_bit)
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


def test_block_radix_sort_key_value():
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

        coop.block.radix_sort_keys(
            thread_keys,
            items_per_thread,
            values=thread_values,
        )

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            keys_out[idx] = thread_keys[i]
            values_out[idx] = thread_values[i]

    h_keys = random_int(items_per_tile, dtype)
    h_values = (h_keys * 7 + 3).astype(dtype)
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


def test_block_radix_sort_key_value_two_phase():
    threads_per_block = 64
    items_per_thread = 2
    items_per_tile = threads_per_block * items_per_thread
    dtype = np.int32

    block_radix_sort = coop.block.radix_sort_keys(
        numba.int32,
        threads_per_block,
        items_per_thread,
        value_dtype=numba.int32,
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

        block_radix_sort(
            thread_keys,
            items_per_thread,
            values=thread_values,
        )

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            keys_out[idx] = thread_keys[i]
            values_out[idx] = thread_values[i]

    h_keys = random_int(items_per_tile, dtype)
    h_values = (h_keys * 5 + 11).astype(dtype)
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


def test_block_radix_sort_blocked_to_striped():
    threads_per_block = 64
    items_per_thread = 2
    items_per_tile = threads_per_block * items_per_thread
    dtype = np.int32

    @cuda.jit
    def kernel(keys_in, keys_out):
        tid = row_major_tid()
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_keys[i] = keys_in[tid * items_per_thread + i]

        coop.block.radix_sort_keys(
            thread_keys,
            items_per_thread,
            blocked_to_striped=True,
        )

        for i in range(items_per_thread):
            keys_out[tid * items_per_thread + i] = thread_keys[i]

    h_keys = random_int(items_per_tile, dtype)
    d_keys = cuda.to_device(h_keys)
    d_keys_out = cuda.device_array(items_per_tile, dtype=dtype)

    kernel[1, threads_per_block](d_keys, d_keys_out)
    cuda.synchronize()

    keys_out = d_keys_out.copy_to_host()
    sorted_keys = np.sort(h_keys)

    expected = np.empty_like(sorted_keys)
    for tid in range(threads_per_block):
        for i in range(items_per_thread):
            expected[tid * items_per_thread + i] = sorted_keys[
                tid + i * threads_per_block
            ]

    np.testing.assert_array_equal(keys_out, expected)


def test_block_radix_sort_decomposer():
    threads_per_block = 64
    items_per_thread = 2
    items_per_tile = threads_per_block * items_per_thread
    component_dtype = np.int32

    @cuda.jit(device=True)
    def decompose(x):
        return (x[0].real, x[0].imag)

    decomposer = coop.Decomposer(decompose, types.UniTuple(types.int32, 2))

    @cuda.jit
    def kernel(real_in, imag_in, real_out, imag_out):
        tid = row_major_tid()
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=complex_type)
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            thread_keys[i] = Complex(real_in[idx], imag_in[idx])

        coop.block.radix_sort_keys(
            thread_keys,
            items_per_thread,
            numba.int32(0),
            numba.int32(64),
            decomposer=decomposer,
        )

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            real_out[idx] = thread_keys[i].real
            imag_out[idx] = thread_keys[i].imag

    h_real = random_int(items_per_tile, component_dtype)
    h_imag = random_int(items_per_tile, component_dtype)
    d_real = cuda.to_device(h_real)
    d_imag = cuda.to_device(h_imag)
    d_real_out = cuda.device_array(items_per_tile, dtype=component_dtype)
    d_imag_out = cuda.device_array(items_per_tile, dtype=component_dtype)

    with pytest.raises(ValueError, match="decomposer"):
        kernel[1, threads_per_block](d_real, d_imag, d_real_out, d_imag_out)


@pytest.mark.parametrize("T", [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize("threads_per_block", [32, 128, 256, 1024, (8, 16), (2, 4, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 3])
def test_block_radix_sort_descending(T, threads_per_block, items_per_thread):
    begin_bit = numba.int32(0)
    end_bit = numba.int32(T.bitwidth)

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
        coop.block.radix_sort_keys_descending(
            thread_data, items_per_thread, begin_bit, end_bit
        )
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


@pytest.mark.parametrize("T", [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize("threads_per_block", [32, 128, 256, 1024, (8, 16), (2, 4, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 3])
def test_block_radix_sort(T, threads_per_block, items_per_thread):
    items_per_tile = (
        threads_per_block * items_per_thread
        if type(threads_per_block) is int
        else reduce(mul, threads_per_block) * items_per_thread
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        coop.block.radix_sort_keys(thread_data, items_per_thread)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
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


def test_block_radix_sort_overloads_work():
    T = numba.int32
    threads_per_block = 128
    items_per_thread = 3
    items_per_tile = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        thread_data = cuda.local.array(shape=items_per_thread, dtype="int32")
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        coop.block.radix_sort_keys(
            thread_data, items_per_thread, numba.int32(0), numba.int32(32)
        )
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input)
    for i in range(items_per_tile):
        assert output[i] == reference[i]


def test_block_radix_sort_mangling():
    return  # TODO Return to linker issue
    threads_per_block = 128
    items_per_thread = 3
    items_per_tile = threads_per_block * items_per_thread

    int_block_radix_sort = coop.block.radix_sort_keys(
        dtype=numba.int32,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
    )
    int_temp_storage_bytes = int_block_radix_sort.temp_storage_bytes

    double_block_radix_sort = coop.block.radix_sort_keys(
        dtype=numba.float64,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
    )
    double_temp_storage_bytes = double_block_radix_sort.temp_storage_bytes

    @cuda.jit
    def kernel(int_input, int_output, double_input, double_output):
        tid = cuda.threadIdx.x
        int_temp_storage = cuda.shared.array(
            shape=int_temp_storage_bytes, dtype="uint8"
        )
        int_thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            int_thread_data[i] = int_input[tid * items_per_thread + i]
        int_block_radix_sort(int_temp_storage, int_thread_data)
        for i in range(items_per_thread):
            int_output[tid * items_per_thread + i] = int_thread_data[i]
        double_temp_storage = cuda.shared.array(
            shape=double_temp_storage_bytes, dtype="uint8"
        )
        double_thread_data = cuda.local.array(
            shape=items_per_thread, dtype=numba.float64
        )
        for i in range(items_per_thread):
            double_thread_data[i] = double_input[tid * items_per_thread + i]
        double_block_radix_sort(double_temp_storage, double_thread_data)
        for i in range(items_per_thread):
            double_output[tid * items_per_thread + i] = double_thread_data[i]

    int_input = random_int(items_per_tile, "int32")
    d_int_input = cuda.to_device(int_input)
    d_int_output = cuda.device_array(items_per_tile, dtype="int32")
    double_input = random_int(items_per_tile, "float64")
    d_double_input = cuda.to_device(double_input)
    d_double_output = cuda.device_array(items_per_tile, dtype="float64")
    kernel[1, threads_per_block](
        d_int_input, d_int_output, d_double_input, d_double_output
    )
    cuda.synchronize()

    int_output = d_int_output.copy_to_host()
    int_reference = sorted(int_input)
    for i in range(items_per_tile):
        assert int_output[i] == int_reference[i]

    double_output = d_double_output.copy_to_host()
    double_reference = sorted(double_input)
    for i in range(items_per_tile):
        assert double_output[i] == double_reference[i]

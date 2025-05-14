# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
from pynvjitlink import patch

import cuda.cooperative.experimental as cudax

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


patch.patch_numba_linker(lto=True)


@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_reduction_of_user_defined_type_without_temp_storage(
    threads_per_block, algorithm
):
    def op(result_ptr, lhs_ptr, rhs_ptr):
        real_value = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_value = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_value, imag_value)

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.reduce(
        dtype=complex_type,
        binary_op=op,
        threads_per_block=threads_per_block,
        algorithm=algorithm,
        methods={
            "construct": Complex.construct,
            "assign": Complex.assign,
        },
    )

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        block_output = block_reduce(
            Complex(input[tid], input[num_threads_per_block + tid])
        )

        if tid == 0:
            output[0] = block_output.real
            output[1] = block_output.imag

    h_input = random_int(2 * num_threads_per_block, "int32")
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(2, dtype="int32")
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = (
        np.sum(h_input[:num_threads_per_block]),
        np.sum(h_input[num_threads_per_block:]),
    )

    assert h_output[0] == h_expected[0]
    assert h_output[1] == h_expected[1]

    sig = (numba.int32[::1], numba.int32[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_reduction_of_user_defined_type(threads_per_block, algorithm):
    def op(result_ptr, lhs_ptr, rhs_ptr):
        real_value = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_value = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_value, imag_value)

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.reduce(
        dtype=complex_type,
        binary_op=op,
        threads_per_block=threads_per_block,
        algorithm=algorithm,
        methods={
            "construct": Complex.construct,
            "assign": Complex.assign,
        },
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        block_output = block_reduce(
            temp_storage,
            Complex(input[tid], input[num_threads_per_block + tid]),
        )

        if tid == 0:
            output[0] = block_output.real
            output[1] = block_output.imag

    h_input = random_int(2 * num_threads_per_block, "int32")
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(2, dtype="int32")
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = (
        np.sum(h_input[:num_threads_per_block]),
        np.sum(h_input[num_threads_per_block:]),
    )

    assert h_output[0] == h_expected[0]
    assert h_output[1] == h_expected[1]

    sig = (numba.int32[::1], numba.int32[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_reduction_of_integral_type(T, threads_per_block, algorithm):
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.reduce(
        dtype=T, binary_op=op, threads_per_block=threads_per_block, algorithm=algorithm
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        block_output = block_reduce(temp_storage, input[tid])

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(num_threads_per_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_reduction_valid(T, threads_per_block, algorithm):
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.reduce(
        dtype=T, binary_op=op, threads_per_block=threads_per_block, algorithm=algorithm
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        block_output = block_reduce(
            temp_storage, input[tid], num_threads_per_block // 2
        )

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(num_threads_per_block, dtype)
    h_input[-1] = 0
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input[: num_threads_per_block // 2])

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [1, 2, 4])
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_reduction_array_local(T, threads_per_block, items_per_thread, algorithm):
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.reduce(
        dtype=T,
        binary_op=op,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        thread_items = cuda.local.array(shape=items_per_thread, dtype=T)

        for i in range(items_per_thread):
            thread_items[i] = input[i * num_threads_per_block + tid]

        if items_per_thread == 1:
            block_output = block_reduce(temp_storage, thread_items[0])
        else:
            block_output = block_reduce(temp_storage, thread_items)

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(items_per_thread * num_threads_per_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [1, 2, 4])
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_reduction_array_global(
    T, threads_per_block, items_per_thread, algorithm
):
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.reduce(
        dtype=T,
        binary_op=op,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")

        if items_per_thread == 1:
            block_output = block_reduce(temp_storage, input[tid])
        else:
            block_input = input[items_per_thread * tid :]
            block_output = block_reduce(temp_storage, block_input)

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(items_per_thread * num_threads_per_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_sum(T, threads_per_block, algorithm):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.sum(
        dtype=T, threads_per_block=threads_per_block, algorithm=algorithm
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        block_output = block_reduce(temp_storage, input[tid])

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(num_threads_per_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_sum_valid(T, threads_per_block, algorithm):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.sum(
        dtype=T, threads_per_block=threads_per_block, algorithm=algorithm
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        block_output = block_reduce(
            temp_storage, input[tid], numba.int32(num_threads_per_block // 2)
        )

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(num_threads_per_block, dtype)
    h_input[-1] = 0
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input[: num_threads_per_block // 2])

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [1, 2, 4])
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_sum_array_local(T, threads_per_block, items_per_thread, algorithm):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.sum(
        dtype=T,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        thread_items = cuda.local.array(shape=items_per_thread, dtype=T)

        for i in range(items_per_thread):
            thread_items[i] = input[i * num_threads_per_block + tid]

        if items_per_thread == 1:
            block_output = block_reduce(temp_storage, thread_items[0])
        else:
            block_output = block_reduce(temp_storage, thread_items)

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(items_per_thread * num_threads_per_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [1, 2, 4])
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_sum_array_global(T, threads_per_block, items_per_thread, algorithm):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_reduce = cudax.block.sum(
        dtype=T,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")

        if items_per_thread == 1:
            block_output = block_reduce(temp_storage, input[tid])
        else:
            block_input = input[items_per_thread * tid :]
            block_output = block_reduce(temp_storage, block_input)

        if tid == 0:
            output[0] = block_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(items_per_thread * num_threads_per_block, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [0, -1, -127])
def test_block_reduce_invalid_items_per_thread(threads_per_block, items_per_thread):
    def op(a, b):
        return a if a < b else b

    with pytest.raises(ValueError):
        cudax.block.reduce(
            dtype=numba.int32,
            binary_op=op,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
        )


@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [0, -1, -127])
def test_block_sum_invalid_items_per_thread(threads_per_block, items_per_thread):
    with pytest.raises(ValueError):
        cudax.block.sum(
            dtype=numba.int32,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
        )


def test_block_reduce_invalid_algorithm():
    def op(a, b):
        return a if a < b else b

    with pytest.raises(ValueError):
        cudax.block.reduce(
            dtype=numba.int32,
            binary_op=op,
            threads_per_block=128,
            algorithm="invalid_algorithm",
        )


def test_block_sum_invalid_algorithm():
    with pytest.raises(ValueError):
        cudax.block.sum(
            dtype=numba.int32,
            threads_per_block=128,
            algorithm="invalid_algorithm",
        )

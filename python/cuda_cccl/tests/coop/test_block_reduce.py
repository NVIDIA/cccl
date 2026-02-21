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
    random_int,
    row_major_tid,
)
from numba import cuda, types

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (4, 8), (2, 4, 8)]
)
@pytest.mark.parametrize(
    "algorithm", ["raking", "raking_commutative_only", "warp_reductions"]
)
def test_block_reduction_of_user_defined_type_without_temp_storage(
    threads_per_block, algorithm
):
    @cuda.jit(device=True)
    def op(result_ptr, lhs_ptr, rhs_ptr):
        real_value = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_value = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_value, imag_value)

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        block_output = coop.block.reduce(
            Complex(input[tid], input[num_threads_per_block + tid]),
            items_per_thread=1,
            binary_op=op,
            algorithm=algorithm,
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
    @cuda.jit(device=True)
    def op(result_ptr, lhs_ptr, rhs_ptr):
        real_value = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_value = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_value, imag_value)

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        block_output = coop.block.reduce(
            Complex(input[tid], input[num_threads_per_block + tid]),
            items_per_thread=1,
            binary_op=op,
            algorithm=algorithm,
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
    @cuda.jit(device=True)
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        block_output = coop.block.reduce(
            input[tid],
            items_per_thread=1,
            binary_op=op,
            algorithm=algorithm,
        )

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
    @cuda.jit(device=True)
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        block_output = coop.block.reduce(
            input[tid],
            items_per_thread=1,
            binary_op=op,
            num_valid=num_threads_per_block // 2,
            algorithm=algorithm,
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
    @cuda.jit(device=True)
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_items = coop.local.array(items_per_thread, dtype=T)

        for i in range(items_per_thread):
            thread_items[i] = input[i * num_threads_per_block + tid]

        block_output = coop.block.reduce(
            thread_items,
            items_per_thread=items_per_thread,
            binary_op=op,
            algorithm=algorithm,
        )

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
    @cuda.jit(device=True)
    def op(a, b):
        return a if a < b else b

    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        if items_per_thread == 1:
            block_output = coop.block.reduce(
                input[tid],
                items_per_thread=1,
                binary_op=op,
                algorithm=algorithm,
            )
        else:
            block_input = input[items_per_thread * tid :]
            block_output = coop.block.reduce(
                block_input,
                items_per_thread=items_per_thread,
                binary_op=op,
                algorithm=algorithm,
            )

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

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        block_output = coop.block.sum(
            input[tid],
            items_per_thread=1,
            algorithm=algorithm,
        )

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

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        block_output = coop.block.sum(
            input[tid],
            items_per_thread=1,
            num_valid=numba.int32(num_threads_per_block // 2),
            algorithm=algorithm,
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

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        thread_items = coop.local.array(items_per_thread, dtype=T)

        for i in range(items_per_thread):
            thread_items[i] = input[i * num_threads_per_block + tid]

        block_output = coop.block.sum(
            thread_items,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
        )

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

    @cuda.jit
    def kernel(input, output):
        tid = row_major_tid()
        if items_per_thread == 1:
            block_output = coop.block.sum(
                input[tid],
                items_per_thread=1,
                algorithm=algorithm,
            )
        else:
            block_input = input[items_per_thread * tid :]
            block_output = coop.block.sum(
                block_input,
                items_per_thread=items_per_thread,
                algorithm=algorithm,
            )

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
        coop.block.reduce(
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
        coop.block.sum(
            dtype=numba.int32,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
        )


def test_block_reduce_invalid_algorithm():
    def op(a, b):
        return a if a < b else b

    with pytest.raises(ValueError):
        coop.block.reduce(
            dtype=numba.int32,
            binary_op=op,
            threads_per_block=128,
            algorithm="invalid_algorithm",
        )


def test_block_sum_invalid_algorithm():
    with pytest.raises(ValueError):
        coop.block.sum(
            dtype=numba.int32,
            threads_per_block=128,
            algorithm="invalid_algorithm",
        )


def test_sum_alignment():
    sum1 = coop.block.sum(
        dtype=types.int32,
        threads_per_block=256,
    )

    sum2 = coop.block.sum(
        dtype=types.float64,
        threads_per_block=256,
    )

    sum3 = coop.block.sum(
        dtype=types.int8,
        threads_per_block=256,
    )

    assert sum1.temp_storage_alignment == 4
    assert sum2.temp_storage_alignment == 8
    assert sum3.temp_storage_alignment == 1


def test_block_reduction():
    # example-begin reduce
    @cuda.jit(device=True)
    def op(a, b):
        return a if a > b else b

    threads_per_block = 128

    @cuda.jit
    def kernel(input, output):
        block_output = coop.block.reduce(
            input[cuda.threadIdx.x],
            items_per_thread=1,
            binary_op=op,
        )

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # example-end reduce

    h_input = np.random.randint(0, 42, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    h_expected = np.max(h_input)

    assert h_output[0] == h_expected


def test_block_sum_api():
    # example-begin sum
    threads_per_block = 128

    @cuda.jit
    def kernel(input, output):
        block_output = coop.block.sum(
            input[cuda.threadIdx.x],
            items_per_thread=1,
        )

        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # example-end sum

    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_per_block


def test_block_reduction_temp_storage_api():
    @cuda.jit(device=True)
    def op_single(a, b):
        return a + b

    @cuda.jit(device=True)
    def op_two(a, b):
        return a + b

    threads_per_block = 128
    block_reduce = coop.block.reduce(
        np.int32,
        threads_per_block,
        op_two,
        items_per_thread=1,
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes
    temp_storage_alignment = block_reduce.temp_storage_alignment

    # example-begin reduce-temp-storage
    @cuda.jit
    def kernel(input, output_single, output_two):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        block_output = coop.block.reduce(
            input[cuda.threadIdx.x],
            items_per_thread=1,
            binary_op=op_single,
            temp_storage=temp_storage,
        )
        block_output_two_phase = block_reduce(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output_single[0] = block_output
            output_two[0] = block_output_two_phase

    # example-end reduce-temp-storage

    h_input = np.random.randint(0, 42, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output_single = cuda.device_array(1, dtype=np.int32)
    d_output_two = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output_single, d_output_two)
    h_output_single = d_output_single.copy_to_host()
    h_output_two = d_output_two.copy_to_host()

    h_expected = np.sum(h_input)
    assert h_output_single[0] == h_expected
    assert h_output_two[0] == h_expected
    assert h_output_single[0] == h_output_two[0]


def test_block_reduce_temp_storage_getitem_sugar():
    @cuda.jit(device=True)
    def op(a, b):
        return a + b

    threads_per_block = 128

    @cuda.jit
    def kernel(input, output):
        temp_storage = coop.TempStorage()
        block_output = coop.block.reduce[temp_storage](
            input[cuda.threadIdx.x],
            binary_op=op,
            items_per_thread=1,
        )
        if cuda.threadIdx.x == 0:
            output[0] = block_output

    h_input = np.random.randint(0, 42, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == np.sum(h_input)


def test_block_sum_temp_storage_api():
    threads_per_block = 128
    block_sum = coop.block.sum(
        np.int32,
        threads_per_block,
        items_per_thread=1,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes
    temp_storage_alignment = block_sum.temp_storage_alignment

    # example-begin sum-temp-storage
    @cuda.jit
    def kernel(input, output_single, output_two):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        block_output = coop.block.sum(
            input[cuda.threadIdx.x],
            items_per_thread=1,
            temp_storage=temp_storage,
        )
        block_output_two_phase = block_sum(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output_single[0] = block_output
            output_two[0] = block_output_two_phase

    # example-end sum-temp-storage

    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output_single = cuda.device_array(1, dtype=np.int32)
    d_output_two = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output_single, d_output_two)
    h_output_single = d_output_single.copy_to_host()
    h_output_two = d_output_two.copy_to_host()

    assert h_output_single[0] == threads_per_block
    assert h_output_two[0] == threads_per_block
    assert h_output_single[0] == h_output_two[0]


def test_block_reduce_two_phase_temp_storage_api():
    @cuda.jit(device=True)
    def op(a, b):
        return a + b

    threads_per_block = 128
    block_reduce = coop.block.reduce(
        np.int32,
        threads_per_block,
        op,
        items_per_thread=1,
    )
    temp_storage_bytes = block_reduce.temp_storage_bytes
    temp_storage_alignment = block_reduce.temp_storage_alignment

    # example-begin reduce-two-phase-temp-storage
    @cuda.jit
    def kernel(input, output):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        block_output = block_reduce(
            input[cuda.threadIdx.x],
            temp_storage=temp_storage,
        )
        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # example-end reduce-two-phase-temp-storage

    h_input = np.random.randint(0, 42, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == np.sum(h_input)


def test_block_sum_two_phase_temp_storage_api():
    threads_per_block = 128
    block_sum = coop.block.sum(
        np.int32,
        threads_per_block,
        items_per_thread=1,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes
    temp_storage_alignment = block_sum.temp_storage_alignment

    # example-begin sum-two-phase-temp-storage
    @cuda.jit
    def kernel(input, output):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        block_output = block_sum(
            input[cuda.threadIdx.x],
            temp_storage=temp_storage,
        )
        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # example-end sum-two-phase-temp-storage

    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_per_block

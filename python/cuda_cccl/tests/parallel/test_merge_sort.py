# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

import cupy as cp
import numba.cuda
import numba.types
import numpy as np
import pytest

import cuda.cccl.parallel.experimental as parallel

DTYPE_LIST = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
]


def random_array(size, dtype, max_value=None) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        if max_value is None:
            max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        if dtype == np.float16:  # Cannot generate float16 directly
            return rng.random(size=size, dtype=np.float32).astype(dtype)
        else:
            return rng.random(size=size, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def type_to_problem_sizes(dtype) -> List[int]:
    if dtype in DTYPE_LIST:
        return [2, 4, 6, 8, 10, 16, 20]
    else:
        raise ValueError("Unsupported dtype")


def merge_sort_device(
    d_in_keys, d_in_items, d_out_keys, d_out_items, op, num_items, stream=None
):
    # Use the new single-phase API with automatic temp storage allocation
    parallel.merge_sort(
        d_in_keys, d_in_items, d_out_keys, d_out_items, op, num_items, stream=stream
    )


def compare_op(lhs, rhs):
    return np.uint8(lhs < rhs)


merge_sort_params = [
    (dt, 2**log_size, parallel.OpKind.LESS if dt == np.float16 else compare_op)
    for dt in DTYPE_LIST
    for log_size in type_to_problem_sizes(dt)
]


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_keys(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)

    d_in_keys = numba.cuda.to_device(h_in_keys)

    merge_sort_device(d_in_keys, None, d_in_keys, None, op, num_items)

    h_out_keys = d_in_keys.copy_to_host()
    h_in_keys.sort()

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_pairs(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)
    h_in_items = random_array(num_items, np.float32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)

    merge_sort_device(d_in_keys, d_in_items, d_in_keys, d_in_items, op, num_items)

    h_out_keys = d_in_keys.copy_to_host()
    h_out_items = d_in_items.copy_to_host()

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_items = np.array(h_in_items)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_items)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_keys_copy(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)
    h_out_keys = np.empty(num_items, dtype=dtype)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_out_keys = numba.cuda.to_device(h_out_keys)

    merge_sort_device(d_in_keys, None, d_out_keys, None, op, num_items)

    h_out_keys = d_out_keys.copy_to_host()
    h_in_keys.sort()

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_pairs_copy(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_items = numba.cuda.to_device(h_out_items)

    merge_sort_device(d_in_keys, d_in_items, d_out_keys, d_out_items, op, num_items)

    h_out_keys = d_out_keys.copy_to_host()
    h_out_items = d_out_items.copy_to_host()

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_items = np.array(h_in_items)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_items)


def test_merge_sort_pairs_struct_type():
    @parallel.gpu_struct
    class key_pair:
        a: np.int16
        b: np.uint64

    @parallel.gpu_struct
    class item_pair:
        a: np.int32
        b: np.float32

    def struct_compare_op(lhs, rhs):
        return np.uint8(lhs.b < rhs.b) if lhs.a == rhs.a else np.uint8(lhs.a < rhs.a)

    num_items = 1000

    a_keys = np.random.randint(0, 100, num_items, dtype=np.int16)
    b_keys = np.random.randint(0, 100, num_items, dtype=np.uint64)

    a_items = np.random.randint(0, 100, num_items, dtype=np.int32)
    b_items = np.random.rand(num_items).astype(np.float32)

    h_in_keys = np.empty(num_items, dtype=key_pair.dtype)
    h_in_items = np.empty(num_items, dtype=item_pair.dtype)

    h_in_keys["a"] = a_keys
    h_in_keys["b"] = b_keys

    h_in_items["a"] = a_items
    h_in_items["b"] = b_items

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_keys = cp.asarray(d_in_keys).view(key_pair.dtype)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_in_items = cp.asarray(d_in_items).view(item_pair.dtype)

    merge_sort_device(
        d_in_keys, d_in_items, d_in_keys, d_in_items, struct_compare_op, num_items
    )

    h_out_keys = d_in_keys.get()
    h_out_items = d_in_items.get()

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_items = np.array(h_in_items)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_items)


def test_merge_sort_keys_complex():
    def compare_complex(lhs, rhs):
        return np.uint8(lhs.real < rhs.real)

    num_items = 10000
    max_value = 20  # To ensure that the stability property is being tested
    real = random_array(num_items, np.int64, max_value)
    imaginary = random_array(num_items, np.int64, max_value)

    h_in_keys = real + 1j * imaginary
    d_in_keys = numba.cuda.to_device(h_in_keys)

    merge_sort_device(d_in_keys, None, d_in_keys, None, compare_complex, num_items)

    h_out_keys = d_in_keys.copy_to_host()
    h_in_keys = h_in_keys[np.argsort(h_in_keys.real, stable=True)]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_keys_copy_iterator_input(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)
    h_out_keys = np.empty(num_items, dtype=dtype)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_out_keys = numba.cuda.to_device(h_out_keys)

    i_input = parallel.CacheModifiedInputIterator(d_in_keys, modifier="stream")

    merge_sort_device(i_input, None, d_out_keys, None, op, num_items)

    h_in_keys.sort()
    h_out_keys = d_out_keys.copy_to_host()

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_pairs_copy_iterator_input(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_items = numba.cuda.to_device(h_out_items)

    i_input_keys = parallel.CacheModifiedInputIterator(d_in_keys, modifier="stream")
    i_input_items = parallel.CacheModifiedInputIterator(d_in_items, modifier="stream")

    merge_sort_device(
        i_input_keys, i_input_items, d_out_keys, d_out_items, op, num_items
    )

    h_out_keys = d_out_keys.copy_to_host()
    h_out_items = d_out_items.copy_to_host()

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_items = np.array(h_in_items)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_items)


def test_merge_sort_with_stream(cuda_stream):
    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)
    num_items = 10000

    with cp_stream:
        h_in_keys = random_array(num_items, np.int32)
        d_in_keys = cp.asarray(h_in_keys)
        d_out_keys = cp.empty_like(d_in_keys)

    merge_sort_device(
        d_in_keys, None, d_out_keys, None, compare_op, num_items, stream=cuda_stream
    )

    got = d_out_keys.get()
    h_in_keys.sort()

    np.testing.assert_array_equal(got, h_in_keys)


def test_merge_sort_well_known_less():
    """Test merge sort with well-known LESS operation."""
    dtype = np.int32

    # Create input keys
    d_in_keys = cp.array([5, 2, 8, 1, 9, 3], dtype=dtype)
    d_out_keys = cp.empty_like(d_in_keys)

    # Run merge sort with well-known LESS operation
    parallel.merge_sort(
        d_in_keys, None, d_out_keys, None, parallel.OpKind.LESS, len(d_in_keys)
    )

    # Check the result is correct
    expected = np.array([1, 2, 3, 5, 8, 9])
    np.testing.assert_equal(d_out_keys.get(), expected)


def test_merge_sort_well_known_greater():
    """Test merge sort with well-known GREATER operation (descending)."""
    dtype = np.int32

    # Create input keys
    d_in_keys = cp.array([5, 2, 8, 1, 9, 3], dtype=dtype)
    d_out_keys = cp.empty_like(d_in_keys)

    # Run merge sort with well-known GREATER operation
    parallel.merge_sort(
        d_in_keys, None, d_out_keys, None, parallel.OpKind.GREATER, len(d_in_keys)
    )

    # Check the result is correct (descending order)
    expected = np.array([9, 8, 5, 3, 2, 1])
    np.testing.assert_equal(d_out_keys.get(), expected)


def test_merge_sort_with_values_well_known():
    """Test merge sort with values using well-known operations."""
    dtype = np.int32

    # Create input keys and values
    d_in_keys = cp.array([3, 1, 4, 2], dtype=dtype)
    d_in_values = cp.array([30, 10, 40, 20], dtype=dtype)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)

    # Run merge sort with well-known LESS operation
    parallel.merge_sort(
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        parallel.OpKind.LESS,
        len(d_in_keys),
    )

    # Check both keys and values are sorted correctly
    expected_keys = np.array([1, 2, 3, 4])
    expected_values = np.array([10, 20, 30, 40])
    np.testing.assert_equal(d_out_keys.get(), expected_keys)
    np.testing.assert_equal(d_out_values.get(), expected_values)

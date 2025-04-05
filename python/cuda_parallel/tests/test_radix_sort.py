# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple

import numba
import numpy as np
import pytest

import cuda.parallel.experimental.algorithms as algorithms

DTYPE_LIST = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]

PROBLEM_SIZES = [2, 12, 24]

SORT_ORDERS = [algorithms.SortOrder.ASCENDING, algorithms.SortOrder.DESCENDING]

DTYPE_SIZE_ORDER = [
    (dt, 2**log_size, order)
    for dt in DTYPE_LIST
    for log_size in PROBLEM_SIZES
    for order in SORT_ORDERS
]


def random_array(size, dtype, max_value=None) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        if max_value is None:
            max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        return rng.random(size=size, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def radix_sort_device(
    d_in_keys,
    d_in_values,
    d_out_keys,
    d_out_values,
    order,
    num_items,
    begin_bit=None,
    end_bit=None,
    stream=None,
):
    radix_sort = algorithms.radix_sort(
        d_in_keys, d_in_values, d_out_keys, d_out_values, order
    )

    temp_storage_size = radix_sort(
        None,
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        num_items,
        begin_bit,
        end_bit,
        stream,
    )
    d_temp_storage = numba.cuda.device_array(
        temp_storage_size, dtype=np.uint8, stream=stream.ptr if stream else 0
    )
    radix_sort(
        d_temp_storage,
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        num_items,
        begin_bit,
        end_bit,
        stream,
    )


def host_sort(h_in_keys, h_in_values, order) -> Tuple:
    if order is algorithms.SortOrder.DESCENDING:
        # We do this for stability. We need to cast to a signed integer to properly negate the keys
        signed_dtype = np.dtype(h_in_keys.dtype.name.replace("uint", "int"))
        argsort = np.argsort(-h_in_keys.astype(signed_dtype), stable=True)
    else:
        argsort = np.argsort(h_in_keys, stable=True)

    h_in_keys = h_in_keys[argsort]
    if h_in_values is not None:
        h_in_values = h_in_values[argsort]

    return h_in_keys, h_in_values


@pytest.mark.parametrize(
    "dtype, num_items, order",
    DTYPE_SIZE_ORDER,
)
def test_radix_sort_keys(dtype, num_items, order):
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_out_keys = np.empty(num_items, dtype=dtype)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_out_keys = numba.cuda.to_device(h_out_keys)

    radix_sort_device(d_in_keys, None, d_out_keys, None, order, num_items)

    h_out_keys = d_out_keys.copy_to_host()

    h_in_keys, _ = host_sort(h_in_keys, None, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize(
    "dtype, num_items, order",
    DTYPE_SIZE_ORDER,
)
def test_radix_sort_pairs(dtype, num_items, order):
    h_in_keys = random_array(num_items, dtype, max_value=2)
    h_in_values = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_values = np.empty(num_items, dtype=np.float32)

    print(h_in_keys)
    print(h_in_values)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_values = numba.cuda.to_device(h_in_values)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_values = numba.cuda.to_device(h_out_values)

    radix_sort_device(
        d_in_keys, d_in_values, d_out_keys, d_out_values, order, num_items
    )

    h_out_keys = d_out_keys.copy_to_host()
    h_out_values = d_out_values.copy_to_host()

    h_in_keys, h_in_values = host_sort(h_in_keys, h_in_values, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_values, h_in_values)


@pytest.mark.parametrize(
    "dtype, num_items, order",
    DTYPE_SIZE_ORDER,
)
def test_radix_sort_keys_double_buffer(dtype, num_items, order):
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_out_keys = np.empty(num_items, dtype=dtype)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_out_keys = numba.cuda.to_device(h_out_keys)

    keys_double_buffer = algorithms.DoubleBuffer(d_in_keys, d_out_keys)

    radix_sort_device(keys_double_buffer, None, None, None, order, num_items)

    h_out_keys = keys_double_buffer.current().copy_to_host()

    h_in_keys, _ = host_sort(h_in_keys, None, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize(
    "dtype, num_items, order",
    DTYPE_SIZE_ORDER,
)
def test_radix_sort_pairs_double_buffer(dtype, num_items, order):
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_in_values = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_values = np.empty(num_items, dtype=np.float32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_values = numba.cuda.to_device(h_in_values)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_values = numba.cuda.to_device(h_out_values)

    keys_double_buffer = algorithms.DoubleBuffer(d_in_keys, d_out_keys)
    values_double_buffer = algorithms.DoubleBuffer(d_in_values, d_out_values)

    radix_sort_device(
        keys_double_buffer, values_double_buffer, None, None, order, num_items
    )

    h_out_keys = keys_double_buffer.current().copy_to_host()
    h_out_values = values_double_buffer.current().copy_to_host()

    h_in_keys, h_in_values = host_sort(h_in_keys, h_in_values, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_values, h_in_values)

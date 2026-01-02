# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
from typing import Tuple

import cupy as cp
import numba
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    DoubleBuffer,
    SortOrder,
)


def get_mark(dt, log_size):
    if log_size < 20:
        return tuple()
    return pytest.mark.large


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

PROBLEM_SIZES = [2, 10, 20]

DTYPE_SIZE = [
    pytest.param(dt, 2**log_size, marks=get_mark(dt, log_size))
    for dt in DTYPE_LIST
    for log_size in PROBLEM_SIZES
]


def random_array(size, dtype, max_value=None) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        if max_value is None:
            max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        return np.random.uniform(low=-10.0, high=10.0, size=size).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def radix_sort_device(
    d_in_keys,
    d_out_keys,
    d_in_values,
    d_out_values,
    order,
    num_items,
    begin_bit=None,
    end_bit=None,
    stream=None,
):
    # Use the new single-phase API with automatic temp storage allocation
    cuda.compute.radix_sort(
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        order,
        num_items,
        begin_bit,
        end_bit,
        stream,
    )


def get_floating_point_keys(array):
    """
    This function computes the keys for floating point types.
    From the cub docs, this is the required behavior:

    For positive floating point values, the sign bit is inverted.
    For negative floating point values, the full key is inverted.
    """
    if array.dtype == np.float32:
        uint_type = np.uint32
        sign_mask = np.uint32(0x80000000)
    elif array.dtype == np.float64:
        uint_type = np.uint64
        sign_mask = np.uint64(0x8000000000000000)

    # Get the binary representation as unsigned integers
    binary = array.copy().view(uint_type)

    # Create masks for positive and negative numbers
    is_positive = array >= 0
    is_negative = ~is_positive

    # For positive numbers: flip the sign bit (leftmost bit)
    binary[is_positive] ^= sign_mask

    # For negative numbers: invert all bits
    binary[is_negative] = ~binary[is_negative]

    return binary


def host_sort(h_in_keys, h_in_values, order, begin_bit=None, end_bit=None) -> Tuple:
    if begin_bit is not None and end_bit is not None:
        num_bits = end_bit - begin_bit
        mask = np.array(((1 << (num_bits)) - 1) << begin_bit, dtype=np.uint64)

        if np.issubdtype(h_in_keys.dtype, np.floating):
            h_in_keys_copy = get_floating_point_keys(h_in_keys)
        else:
            h_in_keys_copy = h_in_keys

        h_in_keys_copy = h_in_keys_copy.astype(np.uint64)
        h_in_keys_copy = (h_in_keys_copy & mask) >> begin_bit
    else:
        h_in_keys_copy = h_in_keys

    if order is SortOrder.DESCENDING:
        # We do this for stability. We need to cast to a signed integer to properly negate the keys.
        signed_dtype = np.dtype(h_in_keys_copy.dtype.name.replace("uint", "int"))
        argsort = np.argsort(-h_in_keys_copy.astype(signed_dtype), stable=True)
    else:
        argsort = np.argsort(h_in_keys_copy, stable=True)

    h_in_keys = h_in_keys[argsort]
    if h_in_values is not None:
        h_in_values = h_in_values[argsort]

    return h_in_keys, h_in_values


@pytest.mark.parametrize(
    "dtype, num_items",
    DTYPE_SIZE,
)
def test_radix_sort_keys(dtype, num_items, monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # Skip sass verification for CC 9.0+ due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if cc_major >= 9:
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    order = SortOrder.ASCENDING
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_out_keys = np.empty(num_items, dtype=dtype)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_out_keys = numba.cuda.to_device(h_out_keys)

    radix_sort_device(d_in_keys, d_out_keys, None, None, order, num_items)

    h_out_keys = d_out_keys.copy_to_host()

    h_in_keys, _ = host_sort(h_in_keys, None, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize(
    "dtype, num_items",
    DTYPE_SIZE,
)
def test_radix_sort_pairs(dtype, num_items, monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    if cc_major >= 9 or np.isdtype(dtype, (np.int8, np.uint8, np.int16, np.uint32)):
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    order = SortOrder.DESCENDING
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_in_values = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_values = np.empty(num_items, dtype=np.float32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_values = numba.cuda.to_device(h_in_values)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_values = numba.cuda.to_device(h_out_values)

    radix_sort_device(
        d_in_keys, d_out_keys, d_in_values, d_out_values, order, num_items
    )

    h_out_keys = d_out_keys.copy_to_host()
    h_out_values = d_out_values.copy_to_host()

    h_in_keys, h_in_values = host_sort(h_in_keys, h_in_values, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_values, h_in_values)


@pytest.mark.parametrize(
    "dtype, num_items",
    DTYPE_SIZE,
)
def test_radix_sort_keys_double_buffer(dtype, num_items, monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # Skip sass verification for CC 9.0+ due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if cc_major >= 9:
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    order = SortOrder.DESCENDING
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_out_keys = np.empty(num_items, dtype=dtype)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_out_keys = numba.cuda.to_device(h_out_keys)

    keys_double_buffer = DoubleBuffer(d_in_keys, d_out_keys)

    radix_sort_device(keys_double_buffer, None, None, None, order, num_items)

    h_out_keys = keys_double_buffer.current().copy_to_host()

    h_in_keys, _ = host_sort(h_in_keys, None, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize(
    "dtype, num_items",
    DTYPE_SIZE,
)
def test_radix_sort_pairs_double_buffer(dtype, num_items, monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # NOTE: int16 failures seen only with NVRTC 13.1:
    if cc_major >= 9 or np.isdtype(dtype, (np.int16, np.uint32)):
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    order = SortOrder.ASCENDING
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_in_values = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_values = np.empty(num_items, dtype=np.float32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_values = numba.cuda.to_device(h_in_values)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_values = numba.cuda.to_device(h_out_values)

    keys_double_buffer = DoubleBuffer(d_in_keys, d_out_keys)
    values_double_buffer = DoubleBuffer(d_in_values, d_out_values)

    radix_sort_device(
        keys_double_buffer, None, values_double_buffer, None, order, num_items
    )

    h_out_keys = keys_double_buffer.current().copy_to_host()
    h_out_values = values_double_buffer.current().copy_to_host()

    h_in_keys, h_in_values = host_sort(h_in_keys, h_in_values, order)

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_values, h_in_values)


# These tests take longer to execute so we reduce the number of test cases
DTYPE_SIZE_BIT_WINDOW = [
    pytest.param(dt, 2**log_size, marks=get_mark(dt, log_size))
    for dt in [np.uint8, np.int16, np.uint32, np.int64, np.float64]
    for log_size in [2, 24]
]


@pytest.mark.parametrize(
    "dtype, num_items",
    DTYPE_SIZE_BIT_WINDOW,
)
def test_radix_sort_pairs_bit_window(dtype, num_items, monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # NOTE: int16 failures seen only with NVRTC 13.1:
    if cc_major >= 9 or np.isdtype(dtype, (np.int16, np.uint32)):
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    order = SortOrder.ASCENDING
    num_bits = dtype().itemsize
    begin_bits = [0, num_bits // 3, 3 * num_bits // 4, num_bits]
    end_bits = [0, num_bits // 3, 3 * num_bits // 4, num_bits]

    for begin_bit, end_bit in itertools.product(begin_bits, end_bits):
        if end_bit < begin_bit:
            continue

        h_in_keys = random_array(num_items, dtype)
        h_in_values = random_array(num_items, np.float32)
        h_out_keys = np.empty(num_items, dtype=dtype)
        h_out_values = np.empty(num_items, dtype=np.float32)

        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_in_values = numba.cuda.to_device(h_in_values)
        d_out_keys = numba.cuda.to_device(h_out_keys)
        d_out_values = numba.cuda.to_device(h_out_values)

        radix_sort_device(
            d_in_keys,
            d_out_keys,
            d_in_values,
            d_out_values,
            order,
            num_items,
            begin_bit,
            end_bit,
        )

        h_out_keys = d_out_keys.copy_to_host()
        h_out_values = d_out_values.copy_to_host()

        h_in_keys, h_in_values = host_sort(
            h_in_keys, h_in_values, order, begin_bit, end_bit
        )

        np.testing.assert_array_equal(h_out_keys, h_in_keys)
        np.testing.assert_array_equal(h_out_values, h_in_values)


@pytest.mark.parametrize(
    "dtype, num_items",
    DTYPE_SIZE_BIT_WINDOW,
)
def test_radix_sort_pairs_double_buffer_bit_window(dtype, num_items, monkeypatch):
    if np.isdtype(dtype, (np.uint8, np.int16, np.uint32)):
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    order = SortOrder.DESCENDING
    num_bits = dtype().itemsize
    begin_bits = [0, num_bits // 3, 3 * num_bits // 4, num_bits]
    end_bits = [0, num_bits // 3, 3 * num_bits // 4, num_bits]

    for begin_bit, end_bit in itertools.product(begin_bits, end_bits):
        if end_bit < begin_bit:
            continue

        h_in_keys = random_array(num_items, dtype)
        h_in_values = random_array(num_items, np.float32)
        h_out_keys = np.empty(num_items, dtype=dtype)
        h_out_values = np.empty(num_items, dtype=np.float32)

        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_in_values = numba.cuda.to_device(h_in_values)
        d_out_keys = numba.cuda.to_device(h_out_keys)
        d_out_values = numba.cuda.to_device(h_out_values)

        keys_double_buffer = DoubleBuffer(d_in_keys, d_out_keys)
        values_double_buffer = DoubleBuffer(d_in_values, d_out_values)

        radix_sort_device(
            keys_double_buffer,
            None,
            values_double_buffer,
            None,
            order,
            num_items,
            begin_bit,
            end_bit,
        )

        h_out_keys = keys_double_buffer.current().copy_to_host()
        h_out_values = values_double_buffer.current().copy_to_host()

        h_in_keys, h_in_values = host_sort(
            h_in_keys, h_in_values, order, begin_bit, end_bit
        )

        np.testing.assert_array_equal(h_out_keys, h_in_keys)
        np.testing.assert_array_equal(h_out_values, h_in_values)


def test_radix_sort_with_stream(cuda_stream):
    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)
    num_items = 10000

    with cp_stream:
        h_in_keys = random_array(num_items, np.int32)
        d_in_keys = cp.asarray(h_in_keys)
        d_out_keys = cp.empty_like(d_in_keys)

    radix_sort_device(d_in_keys, d_out_keys, None, None, SortOrder.ASCENDING, num_items)

    got = d_out_keys.get()
    h_in_keys.sort()

    np.testing.assert_array_equal(got, h_in_keys)


def test_radix_sort(monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # Skip sass verification for CC 9.0+ due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if cc_major >= 9:
        import cuda.compute._cccl_interop as cccl_interop

        monkeypatch.setattr(
            cccl_interop,
            "_check_sass",
            False,
        )

    import cupy as cp
    import numpy as np

    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)

    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)

    # Call single-phase API directly with num_items parameter
    cuda.compute.radix_sort(
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        SortOrder.ASCENDING,
        d_in_keys.size,
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_out_keys)
    h_out_items = cp.asnumpy(d_out_values)

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_values = np.array(h_in_values)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_values)


def test_radix_sort_double_buffer(monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # Skip sass verification for CC 9.0+ due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if cc_major >= 9:
        import cuda.compute._cccl_interop as cccl_interop

        monkeypatch.setattr(
            cccl_interop,
            "_check_sass",
            False,
        )

    import cupy as cp
    import numpy as np

    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)

    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)

    keys_double_buffer = DoubleBuffer(d_in_keys, d_out_keys)
    values_double_buffer = DoubleBuffer(d_in_values, d_out_values)

    # Call single-phase API directly with num_items parameter
    cuda.compute.radix_sort(
        keys_double_buffer,
        None,
        values_double_buffer,
        None,
        SortOrder.ASCENDING,
        d_in_keys.size,
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(keys_double_buffer.current())
    h_out_values = cp.asnumpy(values_double_buffer.current())

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_values = np.array(h_in_values)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_values, h_in_values)

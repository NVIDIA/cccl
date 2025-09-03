# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import cupy as cp
import numba.cuda
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


def get_mark(dt, log_size):
    if log_size < 20:
        return tuple()
    return pytest.mark.large


PROBLEM_SIZES = [2, 8, 16, 22]


def random_array(size, dtype, max_value=None) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        if max_value is None:
            max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        if dtype == np.float16:  # Cannot generate float16 directly
            return rng.random(size=size, dtype=np.float32).astype(dtype)
        return rng.random(size=size, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def unique_by_key_device(
    d_in_keys,
    d_in_items,
    d_out_keys,
    d_out_items,
    d_out_num_selected,
    op,
    num_items,
    stream=None,
):
    # Call single-phase API directly with all parameters including num_items
    parallel.unique_by_key(
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        op,
        num_items,
        stream,
    )


def is_equal_func(lhs, rhs):
    return lhs == rhs


def unique_by_key_host(keys, items, is_equal=is_equal_func):
    # Must implement our own version of unique_by_key since np.unique() returns
    # unique elements across the entire array, while cub::UniqueByKey
    # de-duplicates consecutive keys that are equal.
    if len(keys) == 0:
        return np.empty(0, dtype=keys.dtype), np.empty(0, dtype=items.dtype)

    prev_key = keys[0]
    keys_out = [prev_key]
    items_out = [items[0]]

    for idx, (previous, next) in enumerate(zip(keys, keys[1:])):
        if not is_equal(previous, next):
            keys_out.append(next)

            # add 1 since we are enumerating over pairs
            items_out.append(items[idx + 1])

    return np.array(keys_out), np.array(items_out)


def compare_op(lhs, rhs):
    return np.uint8(lhs == rhs)


unique_by_key_params = [
    pytest.param(
        dt,
        2**log_size,
        parallel.OpKind.EQUAL_TO if dt == np.float16 else compare_op,
        marks=get_mark(dt, log_size),
    )
    for dt in DTYPE_LIST
    for log_size in PROBLEM_SIZES
]


@pytest.mark.parametrize("dtype, num_items, op", unique_by_key_params)
def test_unique_by_key(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)
    h_out_num_selected = np.empty(1, np.int32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_items = numba.cuda.to_device(h_out_items)
    d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

    unique_by_key_device(
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        op,
        num_items,
    )

    h_out_num_selected = d_out_num_selected.copy_to_host()
    num_selected = h_out_num_selected[0]
    h_out_keys = d_out_keys.copy_to_host()[:num_selected]
    h_out_items = d_out_items.copy_to_host()[:num_selected]

    expected_keys, expected_items = unique_by_key_host(h_in_keys, h_in_items)

    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_items, expected_items)


@pytest.mark.parametrize("dtype, num_items, op", unique_by_key_params)
def test_unique_by_key_iterators(dtype, num_items, op, monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # Skip sass verification for CC 9.0+, due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if cc_major >= 9:
        import cuda.cccl.parallel.experimental._cccl_interop

        monkeypatch.setattr(
            cuda.cccl.parallel.experimental._cccl_interop,
            "_check_sass",
            False,
        )

    h_in_keys = random_array(num_items, dtype, max_value=20)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)
    h_out_num_selected = np.empty(1, np.int64)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_items = numba.cuda.to_device(h_out_items)
    d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

    i_in_keys = parallel.CacheModifiedInputIterator(d_in_keys, modifier="stream")
    i_in_items = parallel.CacheModifiedInputIterator(d_in_items, modifier="stream")

    unique_by_key_device(
        i_in_keys,
        i_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        op,
        num_items,
    )

    h_out_num_selected = d_out_num_selected.copy_to_host()
    num_selected = h_out_num_selected[0]
    h_out_keys = d_out_keys.copy_to_host()[:num_selected]
    h_out_items = d_out_items.copy_to_host()[:num_selected]

    expected_keys, expected_items = unique_by_key_host(h_in_keys, h_in_items)

    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_items, expected_items)


def test_unique_by_key_complex():
    def compare_complex(lhs, rhs):
        return np.uint8(lhs.real == rhs.real)

    num_items = 100000
    max_value = 20
    real = random_array(num_items, np.int64, max_value)
    imaginary = random_array(num_items, np.int64, max_value)

    h_in_keys = real + 1j * imaginary
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=h_in_keys.dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)
    h_out_num_selected = np.empty(1, np.int32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_items = numba.cuda.to_device(h_out_items)
    d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

    unique_by_key_device(
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        compare_complex,
        num_items,
    )

    h_out_num_selected = d_out_num_selected.copy_to_host()
    num_selected = h_out_num_selected[0]
    h_out_keys = d_out_keys.copy_to_host()[:num_selected]
    h_out_items = d_out_items.copy_to_host()[:num_selected]

    expected_keys, expected_items = unique_by_key_host(
        h_in_keys, h_in_items, compare_complex
    )

    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_items, expected_items)


def test_unique_by_key_struct_types():
    @parallel.gpu_struct
    class key_pair:
        a: np.int16
        b: np.uint64

    @parallel.gpu_struct
    class item_pair:
        a: np.int32
        b: np.float32

    def struct_compare_op(lhs, rhs):
        return np.uint8((lhs.a == rhs.a) and (lhs.b == rhs.b))

    num_items = 10000

    a_keys = np.random.randint(0, 20, num_items, dtype=np.int16)
    b_keys = np.random.randint(0, 20, num_items, dtype=np.uint64)

    a_items = np.random.randint(0, 20, num_items, dtype=np.int32)
    b_items = np.random.rand(num_items).astype(np.float32)

    h_in_keys = np.empty(num_items, dtype=key_pair.dtype)
    h_in_items = np.empty(num_items, dtype=item_pair.dtype)
    h_out_num_selected = np.empty(1, np.int64)

    h_in_keys["a"] = a_keys
    h_in_keys["b"] = b_keys

    h_in_items["a"] = a_items
    h_in_items["b"] = b_items

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_keys = cp.asarray(d_in_keys).view(key_pair.dtype)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_in_items = cp.asarray(d_in_items).view(item_pair.dtype)

    d_out_keys = cp.empty_like(d_in_keys)
    d_out_items = cp.empty_like(d_in_items)
    d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

    unique_by_key_device(
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        struct_compare_op,
        num_items,
    )

    h_out_num_selected = d_out_num_selected.copy_to_host()
    num_selected = h_out_num_selected[0]
    h_out_keys = d_out_keys.get()[:num_selected]
    h_out_items = d_out_items.get()[:num_selected]

    expected_keys, expected_items = unique_by_key_host(
        h_in_keys,
        h_in_items,
        lambda lhs, rhs: (lhs["a"] == rhs["a"]) and (lhs["b"] == rhs["b"]),
    )

    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_items, expected_items)


def test_unique_by_key_with_stream(cuda_stream):
    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)
    num_items = 10000

    h_in_keys = random_array(num_items, np.int32, max_value=20)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=np.int32)
    h_out_items = np.empty(num_items, dtype=np.float32)
    h_out_num_selected = np.empty(1, np.int32)

    with cp_stream:
        h_in_keys = random_array(num_items, np.int32)
        d_in_keys = cp.asarray(h_in_keys)
        d_in_items = cp.asarray(h_in_items)
        d_out_keys = cp.empty_like(h_out_keys)
        d_out_items = cp.empty_like(h_out_items)
        d_out_num_selected = cp.empty_like(h_out_num_selected)

    unique_by_key_device(
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        compare_op,
        num_items,
        stream=cuda_stream,
    )

    h_out_keys = d_out_keys.get()
    h_out_items = d_out_items.get()
    h_out_num_selected = d_out_num_selected.get()

    num_selected = h_out_num_selected[0]
    h_out_keys = h_out_keys[:num_selected]
    h_out_items = h_out_items[:num_selected]

    expected_keys, expected_items = unique_by_key_host(h_in_keys, h_in_items)

    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_items, expected_items)


def test_unique_by_key_well_known_equal_to():
    """Test unique by key with well-known EQUAL_TO operation."""
    dtype = np.int32

    # Create input keys and values: keys=[1,1,1,2,2,3] values=[10,20,30,40,50,60]
    d_in_keys = cp.array([1, 1, 1, 2, 2, 3], dtype=dtype)
    d_in_values = cp.array([10, 20, 30, 40, 50, 60], dtype=dtype)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)
    d_num_selected = cp.empty(1, dtype=dtype)

    # Run unique by key with well-known EQUAL_TO operation
    parallel.unique_by_key(
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        d_num_selected,
        parallel.OpKind.EQUAL_TO,
        len(d_in_keys),
    )

    # Check the result is correct
    assert d_num_selected.get()[0] == 3  # three unique keys
    expected_keys = [1, 2, 3]
    expected_values = [10, 40, 60]  # first occurrence of each key
    np.testing.assert_equal(d_out_keys.get()[:3], expected_keys)
    np.testing.assert_equal(d_out_values.get()[:3], expected_values)

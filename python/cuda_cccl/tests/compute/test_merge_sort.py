# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import (
    CacheModifiedInputIterator,
    OpKind,
    deserialize,
    gpu_struct,
    make_merge_sort,
    serialize,
)
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

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
    cuda.compute.merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=d_in_items,
        d_out_keys=d_out_keys,
        d_out_values=d_out_items,
        num_items=num_items,
        op=op,
        stream=stream,
    )


def compare_op(lhs, rhs):
    return np.uint8(lhs < rhs)


merge_sort_params = [
    (dt, 2**log_size, OpKind.LESS if dt == np.float16 else compare_op)
    for dt in DTYPE_LIST
    for log_size in type_to_problem_sizes(dt)
]


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_keys(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)

    d_in_keys = DeviceArray.from_numpy(h_in_keys)

    merge_sort_device(d_in_keys, None, d_in_keys, None, op, num_items)

    h_out_keys = d_in_keys.copy_to_host()
    h_in_keys.sort()

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_pairs(dtype, num_items, op, monkeypatch):
    if dtype == np.float16:
        import cuda.compute._cccl_interop

        monkeypatch.setattr(cuda.compute._cccl_interop, "_check_sass", False)

    h_in_keys = random_array(num_items, dtype)
    h_in_items = random_array(num_items, np.float32)

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_items = DeviceArray.from_numpy(h_in_items)

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

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(h_out_keys.shape, h_out_keys.dtype)

    merge_sort_device(d_in_keys, None, d_out_keys, None, op, num_items)

    h_out_keys = d_out_keys.copy_to_host()
    h_in_keys.sort()

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_pairs_copy(dtype, num_items, op, monkeypatch):
    if dtype == np.float16:
        import cuda.compute._cccl_interop

        monkeypatch.setattr(cuda.compute._cccl_interop, "_check_sass", False)

    h_in_keys = random_array(num_items, dtype)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_items = DeviceArray.from_numpy(h_in_items)
    d_out_keys = DeviceArray.empty(h_out_keys.shape, h_out_keys.dtype)
    d_out_items = DeviceArray.empty(h_out_items.shape, h_out_items.dtype)

    merge_sort_device(d_in_keys, d_in_items, d_out_keys, d_out_items, op, num_items)

    h_out_keys = d_out_keys.copy_to_host()
    h_out_items = d_out_items.copy_to_host()

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_items = np.array(h_in_items)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_items)


def test_merge_sort_pairs_struct_type():
    @gpu_struct
    class key_pair:
        a: np.int16
        b: np.uint64

    @gpu_struct
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

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_items = DeviceArray.from_numpy(h_in_items)

    merge_sort_device(
        d_in_keys, d_in_items, d_in_keys, d_in_items, struct_compare_op, num_items
    )

    h_out_keys = d_in_keys.copy_to_host()
    h_out_items = d_in_items.copy_to_host()

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
    d_in_keys = DeviceArray.from_numpy(h_in_keys)

    merge_sort_device(d_in_keys, None, d_in_keys, None, compare_complex, num_items)

    h_out_keys = d_in_keys.copy_to_host()
    h_in_keys = h_in_keys[np.argsort(h_in_keys.real, stable=True)]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_keys_copy_iterator_input(dtype, num_items, op):
    h_in_keys = random_array(num_items, dtype)
    h_out_keys = np.empty(num_items, dtype=dtype)

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(h_out_keys.shape, h_out_keys.dtype)

    i_input = CacheModifiedInputIterator(d_in_keys, modifier="stream")

    merge_sort_device(i_input, None, d_out_keys, None, op, num_items)

    h_in_keys.sort()
    h_out_keys = d_out_keys.copy_to_host()

    np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize("dtype,num_items,op", merge_sort_params)
def test_merge_sort_pairs_copy_iterator_input(dtype, num_items, op, monkeypatch):
    if dtype == np.float16:
        import cuda.compute._cccl_interop

        monkeypatch.setattr(cuda.compute._cccl_interop, "_check_sass", False)

    h_in_keys = random_array(num_items, dtype)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_items = DeviceArray.from_numpy(h_in_items)
    d_out_keys = DeviceArray.empty(h_out_keys.shape, h_out_keys.dtype)
    d_out_items = DeviceArray.empty(h_out_items.shape, h_out_items.dtype)

    i_input_keys = CacheModifiedInputIterator(d_in_keys, modifier="stream")
    i_input_items = CacheModifiedInputIterator(d_in_items, modifier="stream")

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
    num_items = 10000

    h_in_keys = random_array(num_items, np.int32)
    d_in_keys = DeviceArray.from_numpy(h_in_keys, stream=cuda_stream)
    d_out_keys = DeviceArray.empty(h_in_keys.shape, h_in_keys.dtype, stream=cuda_stream)

    merge_sort_device(
        d_in_keys, None, d_out_keys, None, compare_op, num_items, stream=cuda_stream
    )

    got = d_out_keys.copy_to_host(stream=cuda_stream)
    h_in_keys.sort()

    np.testing.assert_array_equal(got, h_in_keys)


def test_merge_sort_well_known_less():
    dtype = np.int32

    h_in_keys = np.array([5, 2, 8, 1, 9, 3], dtype=dtype)
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(h_in_keys.shape, h_in_keys.dtype)

    cuda.compute.merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=None,
        d_out_keys=d_out_keys,
        d_out_values=None,
        num_items=len(h_in_keys),
        op=OpKind.LESS,
    )

    expected = np.array([1, 2, 3, 5, 8, 9])
    np.testing.assert_equal(d_out_keys.copy_to_host(), expected)


def test_merge_sort_well_known_greater():
    dtype = np.int32

    h_in_keys = np.array([5, 2, 8, 1, 9, 3], dtype=dtype)
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(h_in_keys.shape, h_in_keys.dtype)

    cuda.compute.merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=None,
        d_out_keys=d_out_keys,
        d_out_values=None,
        num_items=len(h_in_keys),
        op=OpKind.GREATER,
    )

    expected = np.array([9, 8, 5, 3, 2, 1])
    np.testing.assert_equal(d_out_keys.copy_to_host(), expected)


def test_merge_sort_large_temp_storage_not_negative():
    """Regression test for https://github.com/NVIDIA/cccl/issues/7911.

    temp_storage_bytes was returned as a signed 32-bit int, overflowing
    to a negative value for large inputs requiring >2GB temp storage.
    """
    num_items = 2**28
    dtype = np.int64
    d_in_keys = DeviceArray.empty(num_items, dtype)
    d_out_keys = DeviceArray.empty(num_items, dtype)

    sorter = cuda.compute.make_merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=None,
        d_out_keys=d_out_keys,
        d_out_values=None,
        op=OpKind.LESS,
    )

    temp_storage_bytes = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_in_values=None,
        d_out_keys=d_out_keys,
        d_out_values=None,
        op=OpKind.LESS,
        num_items=num_items,
    )

    assert temp_storage_bytes > 0


def test_merge_sort_with_values_well_known():
    dtype = np.int32

    h_in_keys = np.array([3, 1, 4, 2], dtype=dtype)
    h_in_values = np.array([30, 10, 40, 20], dtype=dtype)
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_values = DeviceArray.from_numpy(h_in_values)
    d_out_keys = DeviceArray.empty(h_in_keys.shape, h_in_keys.dtype)
    d_out_values = DeviceArray.empty(h_in_values.shape, h_in_values.dtype)

    cuda.compute.merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=d_in_values,
        d_out_keys=d_out_keys,
        d_out_values=d_out_values,
        num_items=len(h_in_keys),
        op=OpKind.LESS,
    )

    expected_keys = np.array([1, 2, 3, 4])
    expected_values = np.array([10, 20, 30, 40])
    np.testing.assert_equal(d_out_keys.copy_to_host(), expected_keys)
    np.testing.assert_equal(d_out_values.copy_to_host(), expected_values)


def _run(sorter, *, d_in_keys, d_in_values, d_out_keys, d_out_values, num_items, op):
    bytes_needed = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_in_values=d_in_values,
        d_out_keys=d_out_keys,
        d_out_values=d_out_values,
        num_items=num_items,
        op=op,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    sorter(
        temp_storage=tmp,
        d_in_keys=d_in_keys,
        d_in_values=d_in_values,
        d_out_keys=d_out_keys,
        d_out_values=d_out_values,
        num_items=num_items,
        op=op,
    )


@pytest.mark.serialization
def test_serialize_deserialize_merge_sort_keys_values():
    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_values = DeviceArray.from_numpy(h_in_values)
    d_out_keys = DeviceArray.empty(h_in_keys.shape, h_in_keys.dtype)
    d_out_values = DeviceArray.empty(h_in_values.shape, h_in_values.dtype)

    builder = make_merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=d_in_values,
        d_out_keys=d_out_keys,
        d_out_values=d_out_values,
        op=OpKind.LESS,
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in_keys=d_in_keys,
        d_in_values=d_in_values,
        d_out_keys=d_out_keys,
        d_out_values=d_out_values,
        num_items=h_in_keys.size,
        op=OpKind.LESS,
    )

    # kind="stable" works on all supported NumPy versions; the stable= keyword
    # was only added in NumPy 2.0 and cuda-cccl pins no numpy floor.
    argsort = np.argsort(h_in_keys, kind="stable")
    np.testing.assert_array_equal(d_out_keys.copy_to_host(), h_in_keys[argsort])
    np.testing.assert_array_equal(d_out_values.copy_to_host(), h_in_values[argsort])


@pytest.mark.serialization
def test_serialize_deserialize_merge_sort_keys_only():
    # Keys-only: d_in_values / d_out_values are None, which become "none"
    # iterators — the plain ITER schema members round-trip them fine.
    h_in_keys = np.array([5, 2, 8, 1, 9, 3, 7, 0, 6, 4], dtype="int32")
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(h_in_keys.shape, h_in_keys.dtype)

    builder = make_merge_sort(
        d_in_keys=d_in_keys,
        d_in_values=None,
        d_out_keys=d_out_keys,
        d_out_values=None,
        op=OpKind.LESS,
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in_keys=d_in_keys,
        d_in_values=None,
        d_out_keys=d_out_keys,
        d_out_values=None,
        num_items=h_in_keys.size,
        op=OpKind.LESS,
    )

    np.testing.assert_array_equal(d_out_keys.copy_to_host(), np.sort(h_in_keys))

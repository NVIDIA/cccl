# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import OpKind
from cuda.compute._utils.protocols import (
    compute_c_contiguous_strides_in_bytes,
)
from cuda.compute.iterators import (
    CacheModifiedInputIterator,
    ConstantIterator,
    CountingIterator,
    ReverseIterator,
    TransformIterator,
)


def test_constant_iterator_equality():
    it1 = ConstantIterator(np.int32(0))
    it2 = ConstantIterator(np.int32(0))
    it3 = ConstantIterator(np.int32(1))
    it4 = ConstantIterator(np.int64(0))

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_counting_iterator_equality():
    it1 = CountingIterator(np.int32(0))
    it2 = CountingIterator(np.int32(0))
    it3 = CountingIterator(np.int32(1))
    it4 = CountingIterator(np.int64(0))

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_cache_modified_input_iterator_equality():
    ary1 = DeviceArray.from_numpy(np.asarray([0, 1, 2], dtype="int32"))
    ary2 = DeviceArray.from_numpy(np.asarray([3, 4, 5], dtype="int32"))
    ary3 = DeviceArray.from_numpy(np.asarray([0, 1, 2], dtype="int64"))

    it1 = CacheModifiedInputIterator(ary1, "stream")
    it2 = CacheModifiedInputIterator(ary1, "stream")
    it3 = CacheModifiedInputIterator(ary2, "stream")
    it4 = CacheModifiedInputIterator(ary3, "stream")

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_equality_transform_iterator():
    def op1(x):
        return x

    def op2(x):
        return 2 * x

    def op3(x):
        return x

    it = CountingIterator(np.int32(0))
    it = CountingIterator(np.int32(1))
    it1 = TransformIterator(it, op1)
    it2 = TransformIterator(it, op1)
    it3 = TransformIterator(it, op3)

    assert it1.kind == it2.kind
    # op3 has a different name than op1, so should have a different kind
    assert it1.kind != it3.kind

    ary1 = DeviceArray.from_numpy(np.asarray([0, 1, 2]))
    ary2 = DeviceArray.from_numpy(np.asarray([3, 4, 5]))
    it4 = TransformIterator(ary1, op1)
    it5 = TransformIterator(ary1, op1)
    it6 = TransformIterator(ary1, op2)
    it7 = TransformIterator(ary1, op3)
    it8 = TransformIterator(ary2, op1)

    assert it4.kind == it5.kind == it8.kind
    # op2 has different bytecode, so should have a different kind
    assert it4.kind != it6.kind
    # op3 has a different name than op1, so should have a different kind
    assert it4.kind != it7.kind


def test_reverse_input_iterator_equality():
    ary1 = DeviceArray.from_numpy(np.asarray([0, 1, 2], dtype="int32"))
    ary2 = DeviceArray.from_numpy(np.asarray([3, 4, 5], dtype="int32"))
    ary3 = DeviceArray.from_numpy(np.asarray([0, 1, 2], dtype="int64"))

    it1 = ReverseIterator(ary1)
    it2 = ReverseIterator(ary1)
    it3 = ReverseIterator(ary2)
    it4 = ReverseIterator(ary3)

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_reverse_output_iterator_equality():
    ary1 = DeviceArray.from_numpy(np.asarray([0, 1, 2], dtype="int32"))
    ary2 = DeviceArray.from_numpy(np.asarray([3, 4, 5], dtype="int32"))
    ary3 = DeviceArray.from_numpy(np.asarray([0, 1, 2], dtype="int64"))

    it1 = ReverseIterator(ary1)
    it2 = ReverseIterator(ary1)
    it3 = ReverseIterator(ary2)
    it4 = ReverseIterator(ary3)

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


@pytest.mark.parametrize(
    "shape, itemsize, expected",
    [
        # Basic 1D
        ((5,), 4, (4,)),
        ((10,), 1, (1,)),
        # Basic 2D
        ((2, 3), 4, (12, 4)),
        ((3, 2), 8, (16, 8)),
        # Basic 3D
        ((2, 3, 4), 1, (12, 4, 1)),
        ((2, 3, 4), 2, (24, 8, 2)),
        # Scalars (0D array)
        ((), 4, ()),
        # Shape with a zero-length dimension
        ((0, 3), 4, (12, 4)),
        ((3, 0), 4, (0, 4)),
    ],
)
def test_compute_c_contiguous_strides_in_bytes(shape, itemsize, expected):
    result = compute_c_contiguous_strides_in_bytes(shape, itemsize)
    assert result == expected


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((2, 3), np.int32),
        ((4, 5, 6), np.float64),
        ((10,), np.uint8),
        ((1,), np.float16),
    ],
)
def test_matches_numpy_strides_for_c_contiguous_arrays(shape, dtype):
    arr = np.zeros(shape, dtype=dtype, order="C")
    expected = arr.strides
    result = compute_c_contiguous_strides_in_bytes(shape, dtype().itemsize)
    assert result == expected


def test_transform_iterator_with_lambda():
    """Test TransformIterator with a lambda function."""
    first_item = 10
    num_items = 100

    # Use a lambda function directly with TransformIterator
    transform_it = TransformIterator(
        CountingIterator(np.int32(first_item)), lambda x: x * 2
    )
    h_init = np.array([0], dtype=np.int32)
    d_output = DeviceArray.empty(1, np.int32)

    # Perform reduction on the transformed iterator
    cuda.compute.reduce_into(
        d_in=transform_it,
        d_out=d_output,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    # Expected: sum of (10*2, 11*2, ..., 109*2) = 2 * sum(10..109)
    expected = 2 * sum(range(first_item, first_item + num_items))
    assert d_output.copy_to_host()[0] == expected


def test_transform_iterator_with_zip_iterator():
    """Test TransformIterator wrapping ZipIterator (struct types)."""
    from cuda.compute.iterators import ZipIterator

    # Create a ZipIterator with two int32 arrays
    h_a = np.arange(10, dtype=np.int32)
    h_b = np.arange(100, 110, dtype=np.int32)
    d_a = DeviceArray.from_numpy(h_a)
    d_b = DeviceArray.from_numpy(h_b)

    zip_it = ZipIterator(d_a, d_b)

    # Create a transform that sums the two fields
    # Input is a struct with two int32 fields, output is a single int32
    def sum_fields(pair):
        return pair[0] + pair[1]

    # Create TransformIterator wrapping ZipIterator
    # This tests that cpp_type_from_descriptor handles struct types correctly
    transform_it = TransformIterator(zip_it, sum_fields)

    # Use it in a reduction
    h_init = np.array([0], dtype=np.int32)
    d_output = DeviceArray.empty(1, np.int32)

    cuda.compute.reduce_into(
        d_in=transform_it,
        d_out=d_output,
        num_items=len(h_a),
        op=OpKind.PLUS,
        h_init=h_init,
    )

    result = d_output.copy_to_host()[0]
    expected = (h_a + h_b).sum()

    assert result == expected, f"Expected {expected}, got {result}"

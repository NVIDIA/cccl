# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numba.cuda
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    CountingIterator,
    TransformIterator,
    ZipIterator,
    gpu_struct,
)


@pytest.mark.parametrize("num_items", [10, 1_000, 100_000])
def test_zip_iterator_basic(num_items):
    @gpu_struct
    class Pair:
        first: np.int64
        second: np.float32

    def sum_pairs(p1, p2):
        return Pair(p1[0] + p2[0], p1[1] + p2[1])

    d_input1 = cp.arange(num_items, dtype=np.int64)
    d_input2 = cp.arange(num_items, dtype=np.float32)

    zip_it = ZipIterator(d_input1, d_input2)

    d_output = cp.empty(1, dtype=Pair.dtype)
    h_init = Pair(0, 0.0)

    cuda.compute.reduce_into(zip_it, d_output, sum_pairs, num_items, h_init)

    expected_first = d_input1.sum().get()
    expected_second = d_input2.sum().get()

    result = d_output.get()[0]
    cp.testing.assert_array_equal(result["first"], expected_first)
    cp.testing.assert_allclose(result["second"], expected_second, rtol=1e-6)


@pytest.mark.parametrize("num_items", [10, 1_000, 100_000])
def test_zip_iterator_with_counting_iterator(num_items):
    """Test ZipIterator with two counting iterators."""

    @gpu_struct
    class IndexValuePair:
        index: np.int32
        value: np.int32

    def max_by_value(p1, p2):
        # Return the pair with the larger value
        return p1 if p1[1] > p2[1] else p2

    counting_it = CountingIterator(np.int32(0))
    arr = cp.arange(num_items, dtype=np.int32)

    zip_it = ZipIterator(counting_it, arr)

    d_output = cp.empty(1, dtype=IndexValuePair.dtype)
    h_init = IndexValuePair(-1, -1)

    cuda.compute.reduce_into(zip_it, d_output, max_by_value, num_items, h_init)

    result = d_output.get()[0]

    expected_index = cp.argmax(arr).get()
    expected_value = arr[expected_index].get()

    assert result["index"] == expected_index
    assert result["value"] == expected_value


@pytest.mark.parametrize("num_items", [10, 1_000, 100_000])
def test_zip_iterator_with_counting_iterator_and_transform(num_items):
    @gpu_struct
    class IndexValuePair:
        index: np.int32
        value: np.int64

    def max_by_value(p1, p2):
        return p1 if p1[1] > p2[1] else p2

    counting_it = CountingIterator(np.int32(0))
    arr = cp.arange(num_items, dtype=np.int32)

    def double_op(x):
        return x * 2

    transform_it = TransformIterator(arr, double_op)

    zip_it = ZipIterator(counting_it, transform_it)

    d_output = cp.empty(1, dtype=IndexValuePair.dtype)

    result = d_output.get()[0]
    h_init = IndexValuePair(-1, -1)

    cuda.compute.reduce_into(zip_it, d_output, max_by_value, num_items, h_init)

    result = d_output.get()[0]

    expected_index = cp.argmax(arr).get()
    expected_value = arr[expected_index].get() * 2

    assert result["index"] == expected_index
    assert result["value"] == expected_value


@pytest.mark.parametrize("num_items", [10, 1_000, 100_000])
def test_zip_iterator_n_iterators(num_items):
    """Test generalized ZipIterator with N iterators (3 in this case)."""

    @gpu_struct
    class Triple:
        first: np.int64
        second: np.float32
        third: np.int64

    def sum_triples(t1, t2):
        return Triple(t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2])

    d_input1 = cp.arange(num_items, dtype=np.int64)
    d_input2 = cp.arange(num_items, dtype=np.float32)
    counting_it = CountingIterator(np.int64(10))

    zip_it = ZipIterator(d_input1, d_input2, counting_it)

    d_output = cp.empty(1, dtype=Triple.dtype)
    h_init = Triple(0, 0.0, 0)

    cuda.compute.reduce_into(zip_it, d_output, sum_triples, num_items, h_init)

    result = d_output.get()[0]

    expected_first = d_input1.sum().get()
    expected_second = d_input2.sum().get()
    expected_third = cp.arange(10, 10 + num_items).sum().get()

    cp.testing.assert_array_equal(result["first"], expected_first)
    cp.testing.assert_allclose(result["second"], expected_second, rtol=1e-6)
    cp.testing.assert_array_equal(result["third"], expected_third)


@pytest.mark.parametrize("num_items", [10, 1_000, 100_000])
def test_zip_iterator_single_iterator(num_items):
    """Test ZipIterator with a single iterator."""

    @gpu_struct
    class Single:
        value: np.int64

    def sum_singles(s1, s2):
        return Single(s1[0] + s2[0])

    d_input = cp.arange(num_items, dtype=np.int64)

    zip_it = ZipIterator(d_input)

    d_output = cp.empty(1, dtype=Single.dtype)
    h_init = Single(0)

    cuda.compute.reduce_into(zip_it, d_output, sum_singles, num_items, h_init)

    result = d_output.get()[0]

    expected_value = d_input.sum().get()
    assert result["value"] == expected_value


@pytest.mark.parametrize("num_items", [10, 1_000])
def test_zip_iterator_with_transform(num_items):
    @gpu_struct
    class TransformedPair:
        sum_indices: np.int32
        product_values: np.int64

    def binary_transform(pair1, pair2):
        return TransformedPair(pair1[0] + pair2[0], pair1[1] * pair2[1])

    counting_it1 = CountingIterator(np.int32(0))
    arr1 = cp.arange(num_items, dtype=np.int32)
    zip_it1 = ZipIterator(counting_it1, arr1)

    counting_it2 = CountingIterator(np.int32(0))
    arr2 = cp.arange(num_items, dtype=np.int32)
    zip_it2 = ZipIterator(counting_it2, arr2)

    d_output = cp.empty(num_items, dtype=TransformedPair.dtype)

    cuda.compute.binary_transform(
        zip_it1, zip_it2, d_output, binary_transform, num_items
    )

    result = d_output.get()

    expected_sum_indices = (arr1 + arr2).get()
    expected_product_values = (arr1 * arr2).get()

    for i, result_item in enumerate(result):
        assert result_item["sum_indices"] == expected_sum_indices[i]
        assert result[i]["product_values"] == expected_product_values[i]


@pytest.mark.parametrize("num_items", [10, 1_000])
def test_zip_iterator_with_scan(num_items):
    """Test ZipIterator with scan operations."""

    @gpu_struct
    class Pair:
        first_min: np.int64
        second_min: np.int64

    def min_pairs(p1, p2):
        # p1 is the accumulated result, p2 is the current input
        # Compute running minimums for both arrays
        return Pair(min(p1[0], p2[0]), min(p1[1], p2[1]))

    # Create two randomized arrays to make min operations interesting
    arr1 = cp.random.randint(0, 1000, num_items, dtype=np.int64)
    arr2 = cp.random.randint(0, 1000, num_items, dtype=np.int64)

    zip_it = ZipIterator(arr1, arr2)

    d_output = cp.empty(num_items, dtype=Pair.dtype)
    h_init = Pair(cp.iinfo(np.int64).max, cp.iinfo(np.int64).max)

    cuda.compute.inclusive_scan(zip_it, d_output, min_pairs, h_init, num_items)

    result = d_output.get()

    # Verify the scan operation produces running minimums for both arrays
    expected_first_running_mins = np.minimum.accumulate(arr1.get())
    expected_second_running_mins = np.minimum.accumulate(arr2.get())

    for i, result_item in enumerate(result):
        assert result_item["first_min"] == expected_first_running_mins[i]
        assert result_item["second_min"] == expected_second_running_mins[i]


@pytest.mark.parametrize("num_items", [10, 1000])
def test_output_zip_iterator_with_scan(monkeypatch, num_items):
    """Test ZipIterator as output iterator with scan operations."""

    # Skip SASS check for CC 8.0+ due to LDL/STL CI failure.
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    if cc_major >= 8:
        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    d_in1 = cp.random.randint(0, 1000, num_items, dtype=np.int64)
    d_in2 = cp.random.randint(0, 1000, num_items, dtype=np.int64)

    zip_it = ZipIterator(d_in1, d_in2)

    d_out1 = cp.empty_like(d_in1)
    d_out2 = cp.empty_like(d_in2)

    zip_out_it = ZipIterator(d_out1, d_out2)

    def add_pairs(p1, p2):
        a, b = tuple(p1)
        c, d = tuple(p2)
        return a + c, b + d

    cuda.compute.inclusive_scan(zip_it, zip_out_it, add_pairs, None, num_items)

    in1 = d_in1.get()
    in2 = d_in2.get()
    expected_out1 = np.empty_like(in1)
    expected_out2 = np.empty_like(in2)

    # First element is just the input
    expected_out1[0] = in1[0]
    expected_out2[0] = in2[0]
    for i in range(1, num_items):
        expected_out1[i] = expected_out1[i - 1] + in1[i]
        expected_out2[i] = expected_out2[i - 1] + in2[i]

    np.testing.assert_array_equal(d_out1.get(), expected_out1)
    np.testing.assert_array_equal(d_out2.get(), expected_out2)

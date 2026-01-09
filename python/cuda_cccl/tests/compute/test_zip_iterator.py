# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
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
    """Test ZipIterator with counting iterator and numpy struct dtype as initial value."""

    def max_by_value(p1, p2):
        # Return the pair with the larger value
        return p1 if p1[1] > p2[1] else p2

    counting_it = CountingIterator(np.int32(0))
    arr = cp.arange(num_items, dtype=np.int32)

    zip_it = ZipIterator(counting_it, arr)

    dtype = np.dtype([("index", np.int32), ("value", np.int32)], align=True)
    h_init = np.asarray([(-1, -1)], dtype=dtype)

    d_output = cp.empty(1, dtype=dtype)

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
    import numba.cuda

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
        return p1[0] + p2[0], p1[1] + p2[1]

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


def test_nested_zip_iterators():
    """Test that ZipIterator can be nested inside another ZipIterator.

    This creates a structure like: ZipIterator(ZipIterator(a, b), c)
    which should produce values with a nested structure.
    """

    InnerPair = gpu_struct({"first": np.int32, "second": np.int64})
    OuterTriple = gpu_struct({"inner": InnerPair, "third": np.float32})

    def sum_nested_zips(v1, v2):
        return OuterTriple(
            InnerPair(
                v1.inner.first + v2.inner.first, v1.inner.second + v2.inner.second
            ),
            v1.third + v2.third,
        )

    num_items = 100

    # Create three input arrays
    d_input_a = cp.arange(num_items, dtype=np.int32)
    d_input_b = cp.arange(num_items, dtype=np.int64) * 2
    d_input_c = cp.arange(num_items, dtype=np.float32) * 3.0

    # Create an inner zip iterator combining a and b
    inner_zip = ZipIterator(d_input_a, d_input_b)

    # Create an outer zip iterator combining inner_zip and c
    outer_zip = ZipIterator(inner_zip, d_input_c)

    # Perform reduction
    d_output = cp.empty(1, dtype=OuterTriple.dtype)
    h_init = OuterTriple(InnerPair(0, 0), 0.0)

    cuda.compute.reduce_into(outer_zip, d_output, sum_nested_zips, num_items, h_init)

    result = d_output.get()[0]

    # Calculate expected values
    expected_first = d_input_a.sum().get()
    expected_second = d_input_b.sum().get()
    expected_third = d_input_c.sum().get()

    assert result["inner"]["first"] == expected_first, (
        f"Expected inner.first={expected_first}, got {result['inner']['first']}"
    )
    assert result["inner"]["second"] == expected_second, (
        f"Expected inner.second={expected_second}, got {result['inner']['second']}"
    )
    assert np.isclose(result["third"], expected_third, rtol=1e-5), (
        f"Expected third={expected_third}, got {result['third']}"
    )


# Test deeply nested zip iterators
def test_deeply_nested_zip_iterators():
    """Test 3 levels of nested zip iterators."""
    # outer_zip produces a struct like: {value_0: {value_0: int32, value_1: float32}, value_1: int64}
    # Define matching struct types with our own names
    InnerPair = gpu_struct({"a": np.int32, "b": np.float32})
    OuterPair = gpu_struct({"inner": InnerPair, "c": np.int64})

    def sum_nested_zips(v1, v2):
        return OuterPair(
            InnerPair(v1.inner.a + v2.inner.a, v1.inner.b + v2.inner.b),
            v1.c + v2.c,
        )

    num_items = 100

    d_input_a = cp.arange(num_items, dtype=np.int32)
    d_input_b = cp.arange(num_items, dtype=np.float32)
    d_input_c = cp.arange(num_items, dtype=np.int64)

    inner_zip = ZipIterator(d_input_a, d_input_b)
    outer_zip = ZipIterator(inner_zip, d_input_c)

    d_output = cp.empty(1, dtype=OuterPair.dtype)
    h_init = OuterPair(InnerPair(0, 0.0), 0)

    cuda.compute.reduce_into(outer_zip, d_output, sum_nested_zips, num_items, h_init)

    result = d_output.get()[0]

    # outer_zip produces: {value_0: {value_0: int32, value_1: float32}, value_1: int64}
    # which maps to our OuterPair: {inner: {a: int32, b: float32}, c: int64}
    expected_a = d_input_a.sum().get()  # int32
    expected_b = d_input_b.sum().get()  # float32
    expected_c = d_input_c.sum().get()  # int64

    assert result["inner"]["a"] == expected_a
    assert np.isclose(result["inner"]["b"], expected_b)
    assert result["c"] == expected_c


@pytest.mark.parametrize("num_items", [10, 1000])
@pytest.mark.parametrize(
    "dtype_map",
    [
        {"x": np.float32, "y": np.float32},
        {"x": np.float64, "y": np.float32},
    ],
)
def test_nested_output_zip_iterator_with_scan(monkeypatch, num_items, dtype_map):
    import numba.cuda

    cc_major, _ = numba.cuda.get_current_device().compute_capability
    if cc_major >= 8:
        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    Vec2 = gpu_struct(dtype_map)

    h_in1 = np.zeros(num_items, dtype=Vec2.dtype)
    h_in2 = np.zeros(num_items, dtype=Vec2.dtype)
    for i in range(num_items):
        h_in1[i]["x"] = float(i)
        h_in1[i]["y"] = float(i * 2)
        h_in2[i]["x"] = float(i * 10)
        h_in2[i]["y"] = float(i * 20)

    d_in1 = cp.empty(num_items, dtype=Vec2.dtype)
    d_in2 = cp.empty(num_items, dtype=Vec2.dtype)
    d_in1.set(h_in1)
    d_in2.set(h_in2)

    zip_it = ZipIterator(d_in1, d_in2)

    d_out1 = cp.empty_like(d_in1)
    d_out2 = cp.empty_like(d_in2)

    zip_out_it = ZipIterator(d_out1, d_out2)

    def add_vec2_pairs(v1, v2):
        result1 = (v1[0].x + v2[0].x, v1[0].y + v2[0].y)
        result2 = (v1[1].x + v2[1].x, v1[1].y + v2[1].y)
        return Vec2(result1[0], result1[1]), Vec2(result2[0], result2[1])

    cuda.compute.inclusive_scan(zip_it, zip_out_it, add_vec2_pairs, None, num_items)

    in1 = d_in1.get()
    in2 = d_in2.get()
    expected_out1 = np.empty_like(in1)
    expected_out2 = np.empty_like(in2)

    expected_out1[0] = in1[0]
    expected_out2[0] = in2[0]
    for i in range(1, num_items):
        expected_out1[i]["x"] = expected_out1[i - 1]["x"] + in1[i]["x"]
        expected_out1[i]["y"] = expected_out1[i - 1]["y"] + in1[i]["y"]
        expected_out2[i]["x"] = expected_out2[i - 1]["x"] + in2[i]["x"]
        expected_out2[i]["y"] = expected_out2[i - 1]["y"] + in2[i]["y"]

    np.testing.assert_array_equal(d_out1.get(), expected_out1)
    np.testing.assert_array_equal(d_out2.get(), expected_out2)


def test_zip_iterator_of_transform_iterator_kind():
    arr = cp.arange(10, dtype=np.int64)

    def f(x):
        return x

    def g(x):
        return x + 1

    it1 = ZipIterator(TransformIterator(arr, f))
    it2 = ZipIterator(TransformIterator(arr, g))
    assert it1.kind != it2.kind


def test_caching_zip_iterator():
    # counting iterators with the same value type:
    z1 = ZipIterator(
        CountingIterator(np.int32(0)),
    )
    z2 = ZipIterator(
        CountingIterator(np.int32(0)),
    )
    assert z1.advance is z2.advance
    assert z1.input_dereference is z2.input_dereference

    # counting iterators with different value types:
    z1 = ZipIterator(CountingIterator(np.int32(0)))
    z2 = ZipIterator(CountingIterator(np.int64(0)))
    assert z1.advance is not z2.advance
    assert z1.input_dereference is not z2.input_dereference

    # arrays with the same dtype:
    z1 = ZipIterator(cp.arange(10, dtype=np.int32))
    z2 = ZipIterator(cp.arange(10, dtype=np.int32))
    assert z1.advance is z2.advance
    assert z1.input_dereference is z2.input_dereference
    assert z1.output_dereference is z2.output_dereference

    # arrays with different dtypes:
    z1 = ZipIterator(cp.arange(10, dtype=np.int32))
    z2 = ZipIterator(cp.arange(10, dtype=np.int64))
    assert z1.advance is not z2.advance
    assert z1.input_dereference is not z2.input_dereference
    assert z1.output_dereference is not z2.output_dereference

    # zip of transform iterator with the same op:
    def op(x):
        return x

    z1 = ZipIterator(TransformIterator(cp.arange(10, dtype=np.int32), op))
    z2 = ZipIterator(TransformIterator(cp.arange(10, dtype=np.int32), op))
    assert z1.advance is z2.advance
    assert z1.input_dereference is z2.input_dereference
    # zip of transform iterator with different op:

    def op2(x):
        return x + 1

    z1 = ZipIterator(TransformIterator(cp.arange(10, dtype=np.int32), op))
    z2 = ZipIterator(TransformIterator(cp.arange(10, dtype=np.int32), op2))
    assert z1.advance is not z2.advance
    assert z1.input_dereference is not z2.input_dereference
    # zip of transform iterator with different input iterator:
    z1 = ZipIterator(TransformIterator(cp.arange(10, dtype=np.int32), op))
    z2 = ZipIterator(TransformIterator(cp.arange(10, dtype=np.int64), op))
    assert z1.advance is not z2.advance
    assert z1.input_dereference is not z2.input_dereference

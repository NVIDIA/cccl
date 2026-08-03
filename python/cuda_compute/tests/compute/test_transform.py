# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    deserialize,
    gpu_struct,
    make_binary_transform,
    make_unary_transform,
    serialize,
)


def unary_transform_host(h_input: np.ndarray, op):
    return np.vectorize(op)(h_input)


def unary_transform_device(d_input, d_output, num_items, op, stream=None):
    cuda.compute.unary_transform(
        d_in=d_input, d_out=d_output, op=op, num_items=num_items, stream=stream
    )


def binary_transform_host(h_input1: np.ndarray, h_input2: np.ndarray, op):
    return np.vectorize(op)(h_input1, h_input2)


def binary_transform_device(d_input1, d_input2, d_output, num_items, op, stream=None):
    cuda.compute.binary_transform(
        d_in1=d_input1,
        d_in2=d_input2,
        d_out=d_output,
        op=op,
        num_items=num_items,
        stream=stream,
    )


def test_unary_transform(input_array):
    if input_array.dtype == np.float16:
        pytest.skip("float16 is not supported with custom operators")

    def op(a):
        return a + 1

    h_in = input_array
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    unary_transform_device(d_in, d_out, h_in.size, op)

    got = d_out.copy_to_host()
    expected = unary_transform_host(h_in, op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_binary_transform(input_array):
    if input_array.dtype == np.float16:
        pytest.skip("float16 is not supported with custom operators")

    def op(a, b):
        return a + b

    h_in1 = input_array
    h_in2 = input_array
    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype)

    binary_transform_device(d_in1, d_in2, d_out, h_in1.size, op)

    got = d_out.copy_to_host()
    expected = binary_transform_host(h_in1, h_in2, op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_unary_transform_struct_type():
    import numpy as np

    @gpu_struct
    class MyStruct:
        x: np.int16
        y: np.uint64

    def op(a):
        return MyStruct(a.x * 2, a.y + 10)

    num_values = 10_000

    h_in = np.empty(num_values, dtype=MyStruct.dtype)
    h_in["x"] = np.arange(num_values)
    h_in["y"] = 1
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=op, num_items=h_in.size)

    got = d_out.copy_to_host()

    np.testing.assert_allclose(got["x"], np.arange(num_values) * 2)
    np.testing.assert_allclose(got["y"], np.ones(num_values) + 10)


def test_binary_transform_struct_type():
    import numpy as np

    @gpu_struct
    class MyStruct:
        x: np.int16
        y: np.uint64

    def op(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    num_values = 10_000

    h_in1 = np.empty(num_values, dtype=MyStruct.dtype)
    h_in1["x"] = np.random.randint(0, num_values, num_values, dtype="int16")
    h_in1["y"] = np.random.randint(0, num_values, num_values, dtype="uint64")

    h_in2 = np.empty(num_values, dtype=MyStruct.dtype)
    h_in2["x"] = np.random.randint(0, num_values, num_values, dtype="int16")
    h_in2["y"] = np.random.randint(0, num_values, num_values, dtype="uint64")

    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype)

    cuda.compute.binary_transform(
        d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=op, num_items=h_in1.size
    )

    got = d_out.copy_to_host()

    np.testing.assert_allclose(got["x"], h_in1["x"] + h_in2["x"])
    np.testing.assert_allclose(got["y"], h_in1["y"] + h_in2["y"])


def test_unary_transform_iterator_input():
    def op(a):
        return a + 1

    d_in = CountingIterator(np.int32(0))

    num_items = 1024
    d_out = DeviceArray.empty(num_items, np.int32)

    unary_transform_device(d_in, d_out, num_items, op)

    got = d_out.copy_to_host()
    expected = np.arange(1, num_items + 1, dtype=np.int32)

    np.testing.assert_allclose(expected, got)


def test_binary_transform_iterator_input():
    def op(a, b):
        return a + b

    d_in1 = CountingIterator(np.int32(0))
    d_in2 = CountingIterator(np.int32(1))

    num_items = 1024
    d_out = DeviceArray.empty(num_items, np.int32)

    binary_transform_device(d_in1, d_in2, d_out, num_items, op)

    got = d_out.copy_to_host()
    expected = np.arange(1, 2 * num_items + 1, step=2, dtype=np.int32)

    np.testing.assert_allclose(expected, got)


def test_unary_transform_with_stream(cuda_stream):
    def op(a):
        return a + 1

    n = 10
    h_in = np.arange(n, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype, stream=cuda_stream)

    unary_transform_device(d_in, d_out, n, op, stream=cuda_stream)

    got = d_out.copy_to_host(stream=cuda_stream)
    expected = unary_transform_host(h_in, op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_binary_transform_with_stream(cuda_stream):
    def op(a, b):
        return a + b

    n = 10
    h_in1 = np.arange(n, dtype=np.int32)
    h_in2 = np.arange(n, dtype=np.int32)
    d_in1 = DeviceArray.from_numpy(h_in1, stream=cuda_stream)
    d_in2 = DeviceArray.from_numpy(h_in2, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype, stream=cuda_stream)

    binary_transform_device(d_in1, d_in2, d_out, n, op, stream=cuda_stream)

    got = d_out.copy_to_host(stream=cuda_stream)
    expected = binary_transform_host(h_in1, h_in2, op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_transform_reuse_input_iterator():
    def op(a, b):
        return a + b

    d_in1 = CountingIterator(np.int32(0))
    d_in2 = CountingIterator(np.int32(1))

    num_items = 1024
    d_out = DeviceArray.empty(num_items, np.int32)

    binary_transform_device(d_in1, d_in2, d_out, num_items, op)

    got = d_out.copy_to_host()
    expected = np.arange(1, 2 * num_items + 1, step=2, dtype=np.int32)

    np.testing.assert_allclose(expected, got)

    # Reusing the second input iterator should work.
    # This is to test that the iterator is not modified by LTOIR scrubbing,
    # which is correctly done on a copy of the iterator.
    def op2(a):
        return a + 1

    unary_transform_device(d_in2, d_out, num_items, op2)
    got = d_out.copy_to_host()
    expected = np.arange(1, num_items + 1, dtype=np.int32) + 1

    np.testing.assert_allclose(expected, got)


def test_unary_transform_well_known_negate():
    """Test unary transform with well-known NEGATE operation."""
    dtype = np.int32
    h_input = np.array([1, -2, 3, -4, 5], dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, dtype)

    # Run unary transform with well-known NEGATE operation
    cuda.compute.unary_transform(
        d_in=d_input, d_out=d_output, op=OpKind.NEGATE, num_items=h_input.size
    )

    # Check the result is correct
    expected = np.array([-1, 2, -3, 4, -5])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_unary_transform_well_known_identity():
    """Test unary transform with well-known IDENTITY operation."""
    dtype = np.int32
    h_input = np.array([1, 2, 3, 4, 5], dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, dtype)

    # Run unary transform with well-known IDENTITY operation
    cuda.compute.unary_transform(
        d_in=d_input, d_out=d_output, op=OpKind.IDENTITY, num_items=h_input.size
    )

    # Check the result is correct
    expected = np.array([1, 2, 3, 4, 5])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_unary_transform_well_known_bit_not():
    h_input = np.array([0, 1, -2, 42, -100], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    cuda.compute.unary_transform(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.BIT_NOT,
        num_items=len(d_input),
    )

    expected = np.array([-1, -2, 1, -43, 99], dtype=np.int32)
    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


@pytest.mark.parametrize("dtype", [np.int32, np.float16])
def test_binary_transform_well_known_plus(dtype):
    """Test binary transform with well-known PLUS operation."""
    h_input1 = np.array([1, 2, 3, 4, 5], dtype=dtype)
    h_input2 = np.array([10, 20, 30, 40, 50], dtype=dtype)
    d_input1 = DeviceArray.from_numpy(h_input1)
    d_input2 = DeviceArray.from_numpy(h_input2)
    d_output = DeviceArray.empty(h_input1.shape, dtype)

    # Run binary transform with well-known PLUS operation
    cuda.compute.binary_transform(
        d_in1=d_input1,
        d_in2=d_input2,
        d_out=d_output,
        op=OpKind.PLUS,
        num_items=h_input1.size,
    )

    # Check the result is correct
    expected = np.array([11, 22, 33, 44, 55])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_binary_transform_well_known_multiplies():
    """Test binary transform with well-known MULTIPLIES operation."""
    dtype = np.int32
    h_input1 = np.array([1, 2, 3, 4, 5], dtype=dtype)
    h_input2 = np.array([2, 3, 4, 5, 6], dtype=dtype)
    d_input1 = DeviceArray.from_numpy(h_input1)
    d_input2 = DeviceArray.from_numpy(h_input2)
    d_output = DeviceArray.empty(h_input1.shape, dtype)

    # Run binary transform with well-known MULTIPLIES operation
    cuda.compute.binary_transform(
        d_in1=d_input1,
        d_in2=d_input2,
        d_out=d_output,
        op=OpKind.MULTIPLIES,
        num_items=h_input1.size,
    )

    # Check the result is correct
    expected = np.array([2, 6, 12, 20, 30])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


@pytest.mark.parametrize(
    "op,host_op",
    [
        pytest.param(OpKind.LOGICAL_AND, np.logical_and, id="logical_and"),
        pytest.param(OpKind.LOGICAL_OR, np.logical_or, id="logical_or"),
    ],
)
def test_binary_transform_well_known_logical(op, host_op):
    h_input1 = np.array([True, True, False, False], dtype=np.bool_)
    h_input2 = np.array([True, False, True, False], dtype=np.bool_)
    d_input1 = DeviceArray.from_numpy(h_input1)
    d_input2 = DeviceArray.from_numpy(h_input2)
    d_output = DeviceArray.empty(h_input1.shape, h_input1.dtype)

    cuda.compute.binary_transform(
        d_in1=d_input1,
        d_in2=d_input2,
        d_out=d_output,
        op=op,
        num_items=len(d_input1),
    )

    np.testing.assert_array_equal(d_output.copy_to_host(), host_op(h_input1, h_input2))


def test_unary_transform_struct_type_with_annotations():
    @gpu_struct
    class Point:
        x: np.float32
        y: np.float32

    def scale_op(p: Point) -> Point:
        return Point(p.x * 2.0, p.y * 3.0)

    num_items = 100

    h_in = np.empty(num_items, dtype=Point.dtype)
    h_in["x"] = np.random.rand(num_items).astype(np.float32)
    h_in["y"] = np.random.rand(num_items).astype(np.float32)

    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=scale_op, num_items=num_items
    )

    result = d_out.copy_to_host()

    np.testing.assert_allclose(result["x"], h_in["x"] * 2.0, rtol=1e-5)
    np.testing.assert_allclose(result["y"], h_in["y"] * 3.0, rtol=1e-5)


def test_binary_transform_struct_type_with_annotations():
    @gpu_struct
    class Vec2D:
        x: np.int32
        y: np.int32

    def add_vectors(v1: Vec2D, v2: Vec2D) -> Vec2D:
        return Vec2D(v1.x + v2.x, v1.y + v2.y)

    num_items = 100

    h_in1 = np.empty(num_items, dtype=Vec2D.dtype)
    h_in1["x"] = np.random.randint(-100, 100, num_items, dtype=np.int32)
    h_in1["y"] = np.random.randint(-100, 100, num_items, dtype=np.int32)

    h_in2 = np.empty(num_items, dtype=Vec2D.dtype)
    h_in2["x"] = np.random.randint(-100, 100, num_items, dtype=np.int32)
    h_in2["y"] = np.random.randint(-100, 100, num_items, dtype=np.int32)

    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype)

    cuda.compute.binary_transform(
        d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=add_vectors, num_items=num_items
    )

    result = d_out.copy_to_host()

    np.testing.assert_equal(result["x"], h_in1["x"] + h_in2["x"])
    np.testing.assert_equal(result["y"], h_in1["y"] + h_in2["y"])


def test_unary_transform_stateful_counting():
    """Test unary_transform with state that counts even numbers."""
    from numba import cuda as numba_cuda

    h_in = np.arange(100, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    even_count = DeviceArray.from_numpy(np.zeros(1, dtype=np.int32))

    # Define op that references state as closure
    def count_evens(x):
        if x % 2 == 0:
            numba_cuda.atomic.add(even_count, 0, 1)
        return x * 2

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=count_evens, num_items=h_in.size
    )

    expected_output = h_in * 2
    np.testing.assert_array_equal(d_out.copy_to_host(), expected_output)

    num_evens = int(even_count.copy_to_host()[0])
    assert num_evens == 50  # 0, 2, 4, ..., 98


def test_unary_transform_stateful_state_updates():
    """Test that stateful transform correctly updates state between calls."""
    num_items = 20
    h_in = np.arange(num_items, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    # Create two different thresholds
    threshold_10 = DeviceArray.from_numpy(np.array([10], dtype=np.int32))
    threshold_15 = DeviceArray.from_numpy(np.array([15], dtype=np.int32))

    # Call 1: x + 10
    def add_threshold_10(x):
        return x + threshold_10[0]

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=add_threshold_10, num_items=num_items
    )
    result_1 = d_out.copy_to_host()
    expected_1 = h_in + 10
    np.testing.assert_array_equal(result_1, expected_1)

    # Call 2: x + 15 (different state)
    def add_threshold_15(x):
        return x + threshold_15[0]

    d_out.copy_from_host(np.zeros_like(h_in))
    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=add_threshold_15, num_items=num_items
    )
    result_2 = d_out.copy_to_host()
    expected_2 = h_in + 15
    np.testing.assert_array_equal(result_2, expected_2)

    # Call 3: Back to first threshold (test cache reuse with updated state)
    d_out.copy_from_host(np.zeros_like(h_in))
    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=add_threshold_10, num_items=num_items
    )
    result_3 = d_out.copy_to_host()
    expected_3 = h_in + 10
    np.testing.assert_array_equal(result_3, expected_3)


def test_unary_transform_stateful_multiple_arrays():
    """Test stateful transform with multiple captured arrays."""
    num_items = 10
    h_in = np.arange(num_items, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    # Multiple state arrays
    offset = DeviceArray.from_numpy(np.array([5], dtype=np.int32))
    multiplier = DeviceArray.from_numpy(np.array([2], dtype=np.int32))

    def transform_with_multiple_state(x):
        return (x + offset[0]) * multiplier[0]

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=transform_with_multiple_state, num_items=num_items
    )
    result = d_out.copy_to_host()
    expected = (h_in + 5) * 2
    np.testing.assert_array_equal(result, expected)

    # Update state and verify it works with new values
    offset = DeviceArray.from_numpy(np.array([10], dtype=np.int32))
    multiplier = DeviceArray.from_numpy(np.array([3], dtype=np.int32))

    def transform_with_updated_state(x):
        return (x + offset[0]) * multiplier[0]

    d_out.copy_from_host(np.zeros_like(h_in))
    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=transform_with_updated_state, num_items=num_items
    )
    result = d_out.copy_to_host()
    expected = (h_in + 10) * 3
    np.testing.assert_array_equal(result, expected)


def test_unary_transform_stateful_closure_factory():
    """Test stateful transform with dynamically created closures.

    This test verifies that state arrays captured in closures created by
    a factory function are properly detected and updated on each call.
    """

    def make_adder(arr):
        """Factory that creates functions with different closure-captured arrays."""

        def func(x):
            return x + arr[0]

        return func

    h_in = np.array([0, 1, 2], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    # First call with offset 10
    cuda.compute.unary_transform(
        d_in=d_in,
        d_out=d_out,
        op=make_adder(DeviceArray.from_numpy(np.array([10], dtype=np.int64))),
        num_items=h_in.size,
    )
    np.testing.assert_array_equal(d_out.copy_to_host(), np.array([10, 11, 12]))

    # Multiple calls with different offsets to test state re-detection
    for i in range(5):
        offset = i * 10
        cuda.compute.unary_transform(
            d_in=d_in,
            d_out=d_out,
            op=make_adder(DeviceArray.from_numpy(np.array([offset], dtype=np.int64))),
            num_items=h_in.size,
        )
        expected = np.array([offset, offset + 1, offset + 2])
        np.testing.assert_array_equal(
            d_out.copy_to_host(),
            expected,
            err_msg=f"Failed at iteration {i} with offset {offset}",
        )


def test_unary_transform_with_lambda():
    """Test unary_transform with a lambda function."""
    h_in = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    # Use a lambda function directly
    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=lambda x: x * 2, num_items=h_in.size
    )

    expected = np.array([2, 4, 6, 8, 10], dtype=np.int32)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_binary_transform_with_lambda():
    """Test binary_transform with a lambda function."""
    h_in1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    h_in2 = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype)

    # Use a lambda function directly
    cuda.compute.binary_transform(
        d_in1=d_in1,
        d_in2=d_in2,
        d_out=d_out,
        op=lambda a, b: a + b,
        num_items=h_in1.size,
    )

    expected = np.array([11, 22, 33, 44, 55], dtype=np.int32)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_binary_transform_bool_equal_to():
    h_input1 = np.array([True, False, True, False], dtype=np.bool_)
    h_input2 = np.array([True, True, False, False], dtype=np.bool_)
    d_input1 = DeviceArray.from_numpy(h_input1)
    d_input2 = DeviceArray.from_numpy(h_input2)
    d_output = DeviceArray.empty(h_input1.shape, h_input1.dtype)

    cuda.compute.binary_transform(
        d_in1=d_input1,
        d_in2=d_input2,
        d_out=d_output,
        op=OpKind.EQUAL_TO,
        num_items=h_input1.size,
    )

    expected = np.array([True, False, False, True], dtype=np.bool_)
    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


def test_stateful_transform_same_bytecode_different_sizes():
    """
    Test that stateful op with same bytecode, but referencing arrays
    of different sizes produce the correct result
    """

    def make_op(arr):
        def op(x):
            return x > len(arr)

        return op

    h_in = np.asarray([1, 2, 3])
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, bool)
    op1 = make_op(DeviceArray.empty(1, np.float64))  # len(arr) == 1
    op2 = make_op(DeviceArray.empty(2, np.float64))  # len(arr) == 2

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=op1, num_items=h_in.size)
    np.testing.assert_array_equal(np.asarray([False, True, True]), d_out.copy_to_host())

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=op2, num_items=h_in.size)
    np.testing.assert_array_equal(
        np.asarray([False, False, True]), d_out.copy_to_host()
    )


def test_transform_caching_with_global_np_ufunc():
    # regression test for a case where if multiple, identically named,
    # ops referenced dotted globals like `np.<func>` those
    # ops would all hash to the same value.

    h_in = np.asarray([1.0, 2.0, 3.0])
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    def make_op():
        sin = np.sin

        def op(x):
            return sin(x)

        return op

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=make_op(), num_items=h_in.size
    )
    np.testing.assert_allclose(d_out.copy_to_host(), np.sin(h_in))

    def make_op():
        cos = np.cos

        def op(x):
            return cos(x)

        return op

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=make_op(), num_items=h_in.size
    )
    np.testing.assert_allclose(d_out.copy_to_host(), np.cos(h_in))


def _add_one(a):
    return a + 1


@pytest.mark.serialization
def test_serialize_deserialize_unary_transform_round_trip():
    h_in = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

    builder = make_unary_transform(d_in=d_in, d_out=d_out, op=_add_one)
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    loaded(d_in=d_in, d_out=d_out, op=_add_one, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host(), h_in + 1)


@pytest.mark.serialization
def test_serialize_deserialize_binary_transform_round_trip():
    h_in1 = np.array([1, 2, 3, 4], dtype=np.int32)
    h_in2 = np.array([10, 20, 30, 40], dtype=np.int32)
    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype)

    builder = make_binary_transform(
        d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=OpKind.PLUS
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    loaded(d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=OpKind.PLUS, num_items=h_in1.size)

    np.testing.assert_array_equal(d_out.copy_to_host(), h_in1 + h_in2)

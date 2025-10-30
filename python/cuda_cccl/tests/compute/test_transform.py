# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    gpu_struct,
)


def unary_transform_host(h_input: np.ndarray, op):
    return np.vectorize(op)(h_input)


def unary_transform_device(d_input, d_output, num_items, op, stream=None):
    cuda.compute.unary_transform(d_input, d_output, op, num_items, stream=stream)


def binary_transform_host(h_input1: np.ndarray, h_input2: np.ndarray, op):
    return np.vectorize(op)(h_input1, h_input2)


def binary_transform_device(d_input1, d_input2, d_output, num_items, op, stream=None):
    cuda.compute.binary_transform(
        d_input1, d_input2, d_output, op, num_items, stream=stream
    )


def test_unary_transform(input_array):
    if input_array.dtype == np.float16:
        pytest.skip("float16 is not supported with custom operators")

    def op(a):
        return a + 1

    d_in = input_array
    d_out = cp.empty_like(d_in)

    unary_transform_device(d_in, d_out, len(d_in), op)

    got = d_out.get()
    expected = unary_transform_host(d_in.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_binary_transform(input_array):
    if input_array.dtype == np.float16:
        pytest.skip("float16 is not supported with custom operators")

    def op(a, b):
        return a + b

    d_in1 = input_array
    d_in2 = input_array
    d_out = cp.empty_like(d_in1)

    binary_transform_device(d_in1, d_in2, d_out, len(d_in1), op)

    got = d_out.get()
    expected = binary_transform_host(d_in1.get(), d_in2.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_unary_transform_struct_type():
    import numpy as np

    MyStruct = gpu_struct({"x": np.int16, "y": np.uint64})

    def op(a):
        return MyStruct(a.x * 2, a.y + 10)

    num_values = 10_000

    h_in = np.empty(num_values, dtype=MyStruct.dtype)
    h_in["x"] = np.arange(num_values)
    h_in["y"] = 1
    d_in = cp.empty_like(h_in)

    cp.cuda.runtime.memcpy(
        d_in.data.ptr,
        h_in.__array_interface__["data"][0],
        h_in.nbytes,
        cp.cuda.runtime.memcpyHostToDevice,
    )

    d_out = cp.empty_like(d_in)

    cuda.compute.unary_transform(d_in, d_out, op, len(d_in))

    got = d_out.get()

    np.testing.assert_allclose(got["x"], np.arange(num_values) * 2)
    np.testing.assert_allclose(got["y"], np.ones(num_values) + 10)


def test_binary_transform_struct_type():
    import numpy as np

    MyStruct = gpu_struct({"x": np.int16, "y": np.uint64})

    def op(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    num_values = 10_000

    h_in1 = np.empty(num_values, dtype=MyStruct.dtype)
    h_in1["x"] = np.random.randint(0, num_values, num_values, dtype="int16")
    h_in1["y"] = np.random.randint(0, num_values, num_values, dtype="uint64")

    h_in2 = np.empty(num_values, dtype=MyStruct.dtype)
    h_in2["x"] = np.random.randint(0, num_values, num_values, dtype="int16")
    h_in2["y"] = np.random.randint(0, num_values, num_values, dtype="uint64")

    d_in1 = cp.empty_like(h_in1)
    d_in2 = cp.empty_like(h_in2)

    cp.cuda.runtime.memcpy(
        d_in1.data.ptr,
        h_in1.__array_interface__["data"][0],
        h_in1.nbytes,
        cp.cuda.runtime.memcpyHostToDevice,
    )
    cp.cuda.runtime.memcpy(
        d_in2.data.ptr,
        h_in2.__array_interface__["data"][0],
        h_in2.nbytes,
        cp.cuda.runtime.memcpyHostToDevice,
    )

    d_out = cp.empty_like(d_in1)

    cuda.compute.binary_transform(d_in1, d_in2, d_out, op, len(d_in1))

    got = d_out.get()

    np.testing.assert_allclose(got["x"], h_in1["x"] + h_in2["x"])
    np.testing.assert_allclose(got["y"], h_in1["y"] + h_in2["y"])


def test_unary_transform_iterator_input():
    def op(a):
        return a + 1

    d_in = CountingIterator(np.int32(0))

    num_items = 1024
    d_out = cp.empty(num_items, dtype=np.int32)

    unary_transform_device(d_in, d_out, num_items, op)

    got = d_out.get()
    expected = np.arange(1, num_items + 1, dtype=np.int32)

    np.testing.assert_allclose(expected, got)


def test_binary_transform_iterator_input():
    def op(a, b):
        return a + b

    d_in1 = CountingIterator(np.int32(0))
    d_in2 = CountingIterator(np.int32(1))

    num_items = 1024
    d_out = cp.empty(num_items, dtype=np.int32)

    binary_transform_device(d_in1, d_in2, d_out, num_items, op)

    got = d_out.get()
    expected = np.arange(1, 2 * num_items + 1, step=2, dtype=np.int32)

    np.testing.assert_allclose(expected, got)


def test_unary_transform_with_stream(cuda_stream):
    def op(a):
        return a + 1

    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)

    n = 10

    with cp_stream:
        d_in = cp.arange(n, dtype=np.int32)
        d_out = cp.empty_like(d_in)

    unary_transform_device(d_in, d_out, n, op, stream=cuda_stream)

    got = d_out.get()
    expected = unary_transform_host(d_in.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_binary_transform_with_stream(cuda_stream):
    def op(a, b):
        return a + b

    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)

    n = 10

    with cp_stream:
        d_in1 = cp.arange(n, dtype=np.int32)
        d_in2 = cp.arange(n, dtype=np.int32)
        d_out = cp.empty_like(d_in1)

    binary_transform_device(d_in1, d_in2, d_out, n, op, stream=cuda_stream)

    got = d_out.get()
    expected = binary_transform_host(d_in1.get(), d_in2.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_transform_reuse_input_iterator():
    def op(a, b):
        return a + b

    d_in1 = CountingIterator(np.int32(0))
    d_in2 = CountingIterator(np.int32(1))

    num_items = 1024
    d_out = cp.empty(num_items, dtype=np.int32)

    binary_transform_device(d_in1, d_in2, d_out, num_items, op)

    got = d_out.get()
    expected = np.arange(1, 2 * num_items + 1, step=2, dtype=np.int32)

    np.testing.assert_allclose(expected, got)

    # Reusing the second input iterator should work.
    # This is to test that the iterator is not modified by LTOIR scrubbing,
    # which is correctly done on a copy of the iterator.
    def op2(a):
        return a + 1

    unary_transform_device(d_in2, d_out, num_items, op2)
    got = d_out.get()
    expected = np.arange(1, num_items + 1, dtype=np.int32) + 1

    np.testing.assert_allclose(expected, got)


def test_unary_transform_well_known_negate():
    """Test unary transform with well-known NEGATE operation."""
    dtype = np.int32
    d_input = cp.array([1, -2, 3, -4, 5], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    # Run unary transform with well-known NEGATE operation
    cuda.compute.unary_transform(d_input, d_output, OpKind.NEGATE, len(d_input))

    # Check the result is correct
    expected = np.array([-1, 2, -3, 4, -5])
    np.testing.assert_equal(d_output.get(), expected)


def test_unary_transform_well_known_identity():
    """Test unary transform with well-known IDENTITY operation."""
    dtype = np.int32
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    # Run unary transform with well-known IDENTITY operation
    cuda.compute.unary_transform(d_input, d_output, OpKind.IDENTITY, len(d_input))

    # Check the result is correct
    expected = np.array([1, 2, 3, 4, 5])
    np.testing.assert_equal(d_output.get(), expected)


@pytest.mark.parametrize("dtype", [np.int32, np.float16])
def test_binary_transform_well_known_plus(dtype):
    """Test binary transform with well-known PLUS operation."""
    d_input1 = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_input2 = cp.array([10, 20, 30, 40, 50], dtype=dtype)
    d_output = cp.empty_like(d_input1, dtype=dtype)

    # Run binary transform with well-known PLUS operation
    cuda.compute.binary_transform(
        d_input1, d_input2, d_output, OpKind.PLUS, len(d_input1)
    )

    # Check the result is correct
    expected = np.array([11, 22, 33, 44, 55])
    np.testing.assert_equal(d_output.get(), expected)


def test_binary_transform_well_known_multiplies():
    """Test binary transform with well-known MULTIPLIES operation."""
    dtype = np.int32
    d_input1 = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_input2 = cp.array([2, 3, 4, 5, 6], dtype=dtype)
    d_output = cp.empty_like(d_input1, dtype=dtype)

    # Run binary transform with well-known MULTIPLIES operation
    cuda.compute.binary_transform(
        d_input1, d_input2, d_output, OpKind.MULTIPLIES, len(d_input1)
    )

    # Check the result is correct
    expected = np.array([2, 6, 12, 20, 30])
    np.testing.assert_equal(d_output.get(), expected)


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

    d_in = cp.empty_like(h_in)
    d_in.set(h_in)

    d_out = cp.empty_like(d_in)

    cuda.compute.unary_transform(d_in, d_out, scale_op, num_items)

    result = d_out.get()

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

    d_in1 = cp.empty_like(h_in1)
    d_in1.set(h_in1)

    d_in2 = cp.empty_like(h_in2)
    d_in2.set(h_in2)

    d_out = cp.empty_like(d_in1)

    cuda.compute.binary_transform(d_in1, d_in2, d_out, add_vectors, num_items)

    result = d_out.get()

    np.testing.assert_equal(result["x"], h_in1["x"] + h_in2["x"])
    np.testing.assert_equal(result["y"], h_in1["y"] + h_in2["y"])

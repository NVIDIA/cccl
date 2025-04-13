# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators
from cuda.parallel.experimental.struct import gpu_struct


def unary_transform_host(h_input: np.ndarray, op):
    return np.vectorize(op)(h_input)


def unary_transform_device(d_input, d_output, num_items, op, stream=None):
    transform = algorithms.unary_transform(d_input, d_output, op)
    transform(d_input, d_output, num_items, stream=stream)


def binary_transform_host(h_input1: np.ndarray, h_input2: np.ndarray, op):
    return np.vectorize(op)(h_input1, h_input2)


def binary_transform_device(d_input1, d_input2, d_output, num_items, op, stream=None):
    transform = algorithms.binary_transform(d_input1, d_input2, d_output, op)
    transform(d_input1, d_input2, d_output, num_items, stream=stream)


def test_unary_transform(input_array):
    # example-begin transform-unary
    import numpy as np

    def op(a):
        return a + 1

    d_in = input_array
    d_out = cp.empty_like(d_in)

    unary_transform_device(d_in, d_out, len(d_in), op)

    got = d_out.get()
    expected = unary_transform_host(d_in.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)
    # example-end transform-unary


def test_binary_transform(input_array):
    # example-begin transform-binary
    import numpy as np

    def op(a, b):
        return a + b

    d_in1 = input_array
    d_in2 = input_array
    d_out = cp.empty_like(d_in1)

    binary_transform_device(d_in1, d_in2, d_out, len(d_in1), op)

    got = d_out.get()
    expected = binary_transform_host(d_in1.get(), d_in2.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)
    # example-end transform-binary


@pytest.mark.skip(reason="https://github.com/NVIDIA/numba-cuda/issues/175")
def test_unary_transform_struct_type():
    import numpy as np

    @gpu_struct
    class MyStruct:
        x: np.int16
        y: np.uint64

    def op(a):
        return MyStruct(a.x * 2, a.y + 10)

    num_values = 10_000

    h_in = np.empty(num_values, dtype=MyStruct)
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

    transform = algorithms.unary_transform(d_in, d_out, op)
    transform(d_in, d_out, len(d_in))

    got = d_out.get()

    np.testing.assert_allclose(got["x"], np.arange(num_values) * 2)
    np.testing.assert_allclose(got["y"], np.ones(num_values) + 10)


@pytest.mark.skip(reason="https://github.com/NVIDIA/numba-cuda/issues/175")
def test_binary_transform_struct_type():
    import numpy as np

    @gpu_struct
    class MyStruct:
        x: np.int16
        y: np.uint64

    def op(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    num_values = 10_000

    h_in1 = np.empty(num_values, dtype=MyStruct)
    h_in1["x"] = np.random.randint(0, num_values, num_values, dtype="int16")
    h_in1["y"] = np.random.randint(0, num_values, num_values, dtype="uint64")

    h_in2 = np.empty(num_values, dtype=MyStruct)
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

    transform = algorithms.binary_transform(d_in1, d_in2, d_out, op)
    transform(d_in1, d_in2, d_out, len(d_in1))

    got = d_out.get()

    np.testing.assert_allclose(got["x"], h_in1["x"] + h_in2["x"])
    np.testing.assert_allclose(got["y"], h_in1["y"] + h_in2["y"])


def test_unary_transform_iterator_input():
    def op(a):
        return a + 1

    d_in = iterators.CountingIterator(np.int32(0))

    num_items = 1024
    d_out = cp.empty(num_items, dtype=np.int32)

    unary_transform_device(d_in, d_out, num_items, op)

    got = d_out.get()
    expected = np.arange(1, num_items + 1, dtype=np.int32)

    np.testing.assert_allclose(expected, got)


def test_binary_transform_iterator_input():
    def op(a, b):
        return a + b

    d_in1 = iterators.CountingIterator(np.int32(0))
    d_in2 = iterators.CountingIterator(np.int32(1))

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

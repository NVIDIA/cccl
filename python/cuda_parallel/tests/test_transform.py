# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.parallel.experimental.algorithms as algorithms
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

    h1_in = np.empty(num_values, dtype=MyStruct)
    h1_in["x"] = np.random.randint(0, num_values, num_values, dtype="int16")
    h1_in["y"] = np.random.randint(0, num_values, num_values, dtype="uint64")

    h2_in = np.empty(num_values, dtype=MyStruct)
    h2_in["x"] = np.random.randint(0, num_values, num_values, dtype="int16")
    h2_in["y"] = np.random.randint(0, num_values, num_values, dtype="uint64")

    d1_in = cp.empty_like(h1_in)
    d2_in = cp.empty_like(h2_in)

    cp.cuda.runtime.memcpy(
        d1_in.data.ptr,
        h1_in.__array_interface__["data"][0],
        h1_in.nbytes,
        cp.cuda.runtime.memcpyHostToDevice,
    )
    cp.cuda.runtime.memcpy(
        d2_in.data.ptr,
        h2_in.__array_interface__["data"][0],
        h2_in.nbytes,
        cp.cuda.runtime.memcpyHostToDevice,
    )

    d_out = cp.empty_like(d1_in)

    transform = algorithms.binary_transform(d1_in, d2_in, d_out, op)
    transform(d1_in, d2_in, d_out, len(d1_in))

    got = d_out.get()

    np.testing.assert_allclose(got["x"], h1_in["x"] + h2_in["x"])
    np.testing.assert_allclose(got["y"], h1_in["y"] + h2_in["y"])

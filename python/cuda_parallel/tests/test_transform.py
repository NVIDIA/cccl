# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms


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
    def op(a):
        return a + 1

    d_in = input_array
    d_out = cp.empty_like(d_in)

    unary_transform_device(d_in, d_out, len(d_in), op)

    got = d_out.get()
    expected = unary_transform_host(d_in.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_binary_transform(input_array):
    def op(a, b):
        return a + b

    d_in1 = input_array
    d_in2 = input_array
    d_out = cp.empty_like(d_in1)

    binary_transform_device(d_in1, d_in2, d_out, len(d_in1), op)

    got = d_out.get()
    expected = binary_transform_host(d_in1.get(), d_in2.get(), op)

    np.testing.assert_allclose(expected, got, rtol=1e-5)

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import cupy as cp
import numba.cuda
import numba.types
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators
from cuda.parallel.experimental.struct import gpu_struct


def exclusive_scan_host(h_input: np.ndarray, op, h_init):
    result = h_input.copy()
    result[0] = h_init[0]
    for i in range(1, len(result)):
        result[i] = op(result[i - 1], h_input[i - 1])
    return result


def exclusive_scan_device(d_input, d_output, num_items, op, h_init, stream=None):
    scan = algorithms.scan(d_input, d_output, op, h_init)
    temp_storage_size = scan(None, d_input, d_output, num_items, h_init, stream=stream)
    d_temp_storage = numba.cuda.device_array(
        temp_storage_size, dtype=np.uint8, stream=stream.ptr if stream else 0
    )
    scan(d_temp_storage, d_input, d_output, num_items, h_init, stream=stream)


def test_scan_array_input(input_array):
    def op(a, b):
        return a + b

    d_input = input_array
    dtype = d_input.dtype
    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty(len(d_input), dtype=dtype)

    exclusive_scan_device(d_input, d_output, len(d_input), op, h_init)

    got = d_output.get()
    expected = exclusive_scan_host(d_input.get(), op, h_init)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_scan_iterator_input():
    def op(a, b):
        return a + b

    d_input = iterators.CountingIterator(np.int32(1))
    num_items = 1024
    dtype = np.dtype("int32")
    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty(num_items, dtype=dtype)

    exclusive_scan_device(d_input, d_output, num_items, op, h_init)

    got = d_output.get()
    expected = exclusive_scan_host(np.arange(1, num_items + 1, dtype=dtype), op, h_init)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_scan_struct_type():
    @gpu_struct
    class XY:
        x: np.int32
        y: np.int32

    def op(a, b):
        return XY(a.x + b.x, a.y + b.y)

    d_input = cp.random.randint(0, 256, (10, 2), dtype=np.int32).view(XY.dtype)
    d_output = cp.empty_like(d_input)

    h_init = XY(0, 0)

    exclusive_scan_device(d_input, d_output, len(d_input), op, h_init)

    got = d_output.get()
    expected_x = exclusive_scan_host(
        d_input.get()["x"], lambda a, b: a + b, np.asarray([h_init.x])
    )
    expected_y = exclusive_scan_host(
        d_input.get()["y"], lambda a, b: a + b, np.asarray([h_init.y])
    )

    np.testing.assert_allclose(expected_x, got["x"], rtol=1e-5)
    np.testing.assert_allclose(expected_y, got["y"], rtol=1e-5)


def test_scan_with_stream(cuda_stream):
    def op(a, b):
        return a + b

    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)

    with cp_stream:
        d_input = cp.random.randint(0, 256, 1024, dtype=np.int32)
        d_output = cp.empty_like(d_input)

    h_init = np.array([42], dtype=np.int32)

    exclusive_scan_device(
        d_input, d_output, len(d_input), op, h_init, stream=cuda_stream
    )

    got = d_output.get()
    expected = exclusive_scan_host(d_input.get(), op, h_init)

    np.testing.assert_allclose(expected, got, rtol=1e-5)

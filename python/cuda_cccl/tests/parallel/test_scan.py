# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import cupy as cp
import numba.cuda
import numba.types
import numpy as np
import pytest

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.cccl.parallel.experimental.iterators as iterators
from cuda.cccl.parallel.experimental.struct import gpu_struct


def scan_host(h_input: np.ndarray, op, h_init, force_inclusive):
    result = h_input.copy()
    if force_inclusive:
        result[0] = op(h_init[0], result[0])
    else:
        result[0] = h_init[0]

    for i in range(1, len(result)):
        if force_inclusive:
            result[i] = op(result[i - 1], h_input[i])
        else:
            result[i] = op(result[i - 1], h_input[i - 1])
    return result


def scan_device(d_input, d_output, num_items, op, h_init, force_inclusive, stream=None):
    scan_algorithm = (
        algorithms.inclusive_scan if force_inclusive else algorithms.exclusive_scan
    )
    scan = scan_algorithm(d_input, d_output, op, h_init)
    temp_storage_size = scan(None, d_input, d_output, num_items, h_init, stream=stream)
    d_temp_storage = numba.cuda.device_array(
        temp_storage_size, dtype=np.uint8, stream=stream.ptr if stream else 0
    )
    scan(d_temp_storage, d_input, d_output, num_items, h_init, stream=stream)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_scan_array_input(force_inclusive, input_array, monkeypatch):
    # Skip sass verification if input is complex
    # as LDL/STL instructions are emitted for complex types.
    if np.issubdtype(input_array.dtype, np.complexfloating):
        import cuda.cccl.parallel.experimental._cccl_interop

        monkeypatch.setattr(
            cuda.cccl.parallel.experimental._cccl_interop,
            "_check_sass",
            False,
        )

    def op(a, b):
        return a + b

    dtype = input_array.dtype
    is_short_dtype = dtype.itemsize < 16
    # for small range data types make input small to assure that
    # accumulation does not overflow
    d_input = input_array[:31] if is_short_dtype else input_array

    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty_like(d_input)

    scan_device(d_input, d_output, len(d_input), op, h_init, force_inclusive)

    got = d_output.get()
    expected = scan_host(d_input.get(), op, h_init, force_inclusive)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_scan_iterator_input(force_inclusive):
    def op(a, b):
        return a + b

    d_input = iterators.CountingIterator(np.int32(1))
    num_items = 1024
    dtype = np.dtype("int32")
    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty(num_items, dtype=dtype)

    scan_device(d_input, d_output, num_items, op, h_init, force_inclusive)

    got = d_output.get()
    expected = scan_host(
        np.arange(1, num_items + 1, dtype=dtype), op, h_init, force_inclusive
    )

    np.testing.assert_allclose(expected, got, rtol=1e-5)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_scan_reverse_counting_iterator_input(force_inclusive):
    def op(a, b):
        return a + b

    num_items = 1024
    d_input = iterators.ReverseInputIterator(
        iterators.CountingIterator(np.int32(num_items))
    )
    dtype = np.dtype("int32")
    h_init = np.array([0], dtype=dtype)
    d_output = cp.empty(num_items, dtype=dtype)

    scan_device(d_input, d_output, num_items, op, h_init, force_inclusive)

    got = d_output.get()
    expected = scan_host(
        np.arange(num_items, 0, -1, dtype=dtype), op, h_init, force_inclusive
    )

    np.testing.assert_allclose(expected, got, rtol=1e-5)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
@pytest.mark.no_verify_sass(reason="LDL/STL instructions emitted for this test.")
def test_scan_struct_type(force_inclusive):
    @gpu_struct
    class XY:
        x: np.int32
        y: np.int32

    def op(a, b):
        return XY(a.x + b.x, a.y + b.y)

    d_input = cp.random.randint(0, 256, (10, 2), dtype=np.int32).view(XY.dtype)
    d_output = cp.empty_like(d_input)

    h_init = XY(0, 0)

    scan_device(d_input, d_output, len(d_input), op, h_init, force_inclusive)

    got = d_output.get()
    expected_x = scan_host(
        d_input.get()["x"], lambda a, b: a + b, np.asarray([h_init.x]), force_inclusive
    )
    expected_y = scan_host(
        d_input.get()["y"], lambda a, b: a + b, np.asarray([h_init.y]), force_inclusive
    )

    np.testing.assert_allclose(expected_x, got["x"], rtol=1e-5)
    np.testing.assert_allclose(expected_y, got["y"], rtol=1e-5)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_scan_with_stream(force_inclusive, cuda_stream):
    def op(a, b):
        return a + b

    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)

    with cp_stream:
        d_input = cp.random.randint(0, 256, 1024, dtype=np.int32)
        d_output = cp.empty_like(d_input)

    h_init = np.array([42], dtype=np.int32)

    scan_device(
        d_input, d_output, len(d_input), op, h_init, force_inclusive, stream=cuda_stream
    )

    got = d_output.get()
    expected = scan_host(d_input.get(), op, h_init, force_inclusive)

    np.testing.assert_allclose(expected, got, rtol=1e-5)

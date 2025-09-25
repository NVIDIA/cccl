# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import cupy as cp
import numba.cuda
import numba.types
import numpy as np
import pytest

import cuda.cccl.parallel.experimental as parallel


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
        parallel.inclusive_scan if force_inclusive else parallel.exclusive_scan
    )
    # Call single-phase API directly with all parameters including num_items
    scan_algorithm(d_input, d_output, op, h_init, num_items, stream)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_scan_array_input(force_inclusive, input_array, monkeypatch):
    cc_major, _ = numba.cuda.get_current_device().compute_capability
    # Skip sass verification if input is complex
    # as LDL/STL instructions are emitted for complex types.
    # Also skip for CC 9.0+, due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if np.issubdtype(input_array.dtype, np.complexfloating) or cc_major >= 9:
        import cuda.cccl.parallel.experimental._cccl_interop

        monkeypatch.setattr(
            cuda.cccl.parallel.experimental._cccl_interop,
            "_check_sass",
            False,
        )

    def op(a, b):
        return a + b

    dtype = input_array.dtype

    if dtype == np.float16:
        reduce_op = parallel.OpKind.PLUS
    else:
        reduce_op = op

    is_short_dtype = dtype.itemsize < 16
    # for small range data types make input small to assure that
    # accumulation does not overflow
    d_input = input_array[:31] if is_short_dtype else input_array

    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty_like(d_input)

    scan_device(d_input, d_output, len(d_input), reduce_op, h_init, force_inclusive)

    got = d_output.get()
    expected = scan_host(d_input.get(), op, h_init, force_inclusive)

    if np.isdtype(dtype, ("real floating", "complex floating")):
        real_dt = np.finfo(dtype).dtype
        eps = np.finfo(real_dt).eps
        rtol = 82 * eps
        np.testing.assert_allclose(expected, got, rtol=rtol)
    else:
        np.testing.assert_array_equal(expected, got)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_scan_iterator_input(force_inclusive):
    def op(a, b):
        return a + b

    d_input = parallel.CountingIterator(np.int32(1))
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
    d_input = parallel.ReverseIterator(parallel.CountingIterator(np.int32(num_items)))
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
    @parallel.gpu_struct
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


def test_exclusive_scan_well_known_plus():
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    parallel.exclusive_scan(
        d_input, d_output, parallel.OpKind.PLUS, h_init, d_input.size
    )

    expected = np.array([0, 1, 3, 6, 10])
    np.testing.assert_equal(d_output.get(), expected)


def test_inclusive_scan_well_known_plus():
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    parallel.inclusive_scan(
        d_input, d_output, parallel.OpKind.PLUS, h_init, d_input.size
    )

    expected = np.array([1, 3, 6, 10, 15])
    np.testing.assert_equal(d_output.get(), expected)


@pytest.mark.xfail(
    reason="CCCL_MAXIMUM well-known operation fails with NVRTC compilation error in C++ library"
)
def test_exclusive_scan_well_known_maximum():
    dtype = np.int32
    h_init = np.array([1], dtype=dtype)
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    parallel.exclusive_scan(
        d_input, d_output, parallel.OpKind.MAXIMUM, h_init, d_input.size
    )

    expected = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    np.testing.assert_equal(d_output.get(), expected)


def test_scan_transform_output_iterator(floating_array):
    """Test scan with TransformOutputIterator."""
    dtype = floating_array.dtype
    h_init = np.array([0], dtype=dtype)

    # Use the floating_array fixture which provides random floating-point data of size 1000
    d_input = floating_array
    d_output = cp.empty_like(d_input, dtype=dtype)

    def square(x: dtype) -> dtype:
        return x * x

    d_out_it = parallel.TransformOutputIterator(d_output, square)

    parallel.inclusive_scan(
        d_input, d_out_it, parallel.OpKind.PLUS, h_init, d_input.size
    )

    expected = cp.cumsum(d_input) ** 2
    # Use more lenient tolerance for float32 due to precision differences
    if dtype == np.float32:
        np.testing.assert_allclose(d_output.get(), expected.get(), atol=1e-4, rtol=1e-4)
    else:
        np.testing.assert_allclose(d_output.get(), expected.get(), atol=1e-6)


def test_exclusive_scan_max():
    def max_op(a, b):
        return max(a, b)

    h_init = np.array([1], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    parallel.exclusive_scan(d_input, d_output, max_op, h_init, d_input.size)

    expected = np.asarray([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    np.testing.assert_equal(d_output.get(), expected)


def test_inclusive_scan_add():
    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    parallel.inclusive_scan(d_input, d_output, add_op, h_init, d_input.size)

    expected = np.asarray([-5, -5, -3, -6, -4, 0, 0, -1, 1, 9])
    np.testing.assert_equal(d_output.get(), expected)


def test_reverse_input_iterator():
    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")
    reverse_it = parallel.ReverseIterator(d_input)

    parallel.inclusive_scan(reverse_it, d_output, add_op, h_init, len(d_input))

    # Check the result is correct
    expected = np.asarray([8, 10, 9, 9, 13, 15, 12, 14, 14, 9])
    np.testing.assert_equal(d_output.get(), expected)


def test_reverse_output_iterator():
    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")
    reverse_it = parallel.ReverseIterator(d_output)

    parallel.inclusive_scan(d_input, reverse_it, add_op, h_init, len(d_input))

    expected = np.asarray([9, 1, -1, 0, 0, -4, -6, -3, -5, -5])
    np.testing.assert_equal(d_output.get(), expected)

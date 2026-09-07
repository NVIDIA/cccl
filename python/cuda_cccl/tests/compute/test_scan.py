# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numpy as np
import pytest
from _utils.device_array import DeviceArray, get_compute_capability

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    ReverseIterator,
    TransformOutputIterator,
    clear_all_caches,
    deserialize,
    exclusive_scan,
    gpu_struct,
    make_exclusive_scan,
    make_inclusive_scan,
    serialize,
)
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer


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
        cuda.compute.inclusive_scan if force_inclusive else cuda.compute.exclusive_scan
    )
    scan_algorithm(
        d_in=d_input,
        d_out=d_output,
        op=op,
        init_value=h_init,
        num_items=num_items,
        stream=stream,
    )


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_scan_array_input(force_inclusive, input_array, monkeypatch):
    cc_major, _ = get_compute_capability()
    # Skip sass verification if input is complex
    # as LDL/STL instructions are emitted for complex types.
    # Also skip for:
    # * uint8-True
    # * int8-True
    # * float64-False
    # Also skip for CC 9.0+, due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if (
        np.issubdtype(input_array.dtype, np.complexfloating)
        or (force_inclusive and np.isdtype(input_array.dtype, (np.uint8, np.int8)))
        or (not force_inclusive and input_array.dtype == np.float64)
        or cc_major >= 9
    ):
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    def op(a, b):
        return a + b

    dtype = input_array.dtype

    if dtype == np.float16:
        reduce_op = OpKind.PLUS
    else:
        reduce_op = op

    is_short_dtype = dtype.itemsize < 16
    # for small range data types make input small to assure that
    # accumulation does not overflow
    h_input = input_array[:31] if is_short_dtype else input_array
    d_input = DeviceArray.from_numpy(h_input)

    h_init = np.array([42], dtype=dtype)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    scan_device(d_input, d_output, h_input.size, reduce_op, h_init, force_inclusive)

    got = d_output.copy_to_host()
    expected = scan_host(h_input, op, h_init, force_inclusive)

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

    d_input = CountingIterator(np.int32(1))
    num_items = 1024
    dtype = np.dtype("int32")
    h_init = np.array([42], dtype=dtype)
    d_output = DeviceArray.empty(num_items, dtype)

    scan_device(d_input, d_output, num_items, op, h_init, force_inclusive)

    got = d_output.copy_to_host()
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
    d_input = ReverseIterator(CountingIterator(np.int32(num_items)))
    dtype = np.dtype("int32")
    h_init = np.array([0], dtype=dtype)
    d_output = DeviceArray.empty(num_items, dtype)

    scan_device(d_input, d_output, num_items, op, h_init, force_inclusive)

    got = d_output.copy_to_host()
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

    h_input = np.random.randint(0, 256, (10, 2), dtype=np.int32).view(XY.dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    h_init = XY(0, 0)

    scan_device(d_input, d_output, len(h_input), op, h_init, force_inclusive)

    got = d_output.copy_to_host()
    expected_x = scan_host(
        h_input["x"], lambda a, b: a + b, np.asarray([h_init.x]), force_inclusive
    )
    expected_y = scan_host(
        h_input["y"], lambda a, b: a + b, np.asarray([h_init.y]), force_inclusive
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

    h_input = np.random.randint(0, 256, 1024, dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input, stream=cuda_stream)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype, stream=cuda_stream)

    h_init = np.array([42], dtype=np.int32)

    scan_device(
        d_input, d_output, h_input.size, op, h_init, force_inclusive, stream=cuda_stream
    )

    got = d_output.copy_to_host(stream=cuda_stream)
    expected = scan_host(h_input, op, h_init, force_inclusive)

    np.testing.assert_allclose(expected, got, rtol=1e-5)


def test_exclusive_scan_well_known_plus():
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    h_input = np.array([1, 2, 3, 4, 5], dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, dtype)

    cuda.compute.exclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.PLUS,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.array([0, 1, 3, 6, 10])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_inclusive_scan_well_known_plus(monkeypatch):
    cc_major, _ = get_compute_capability()
    # Skip SASS check for CC 9.0+, due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if cc_major >= 9:
        import cuda.compute._cccl_interop as cccl_interop

        monkeypatch.setattr(
            cccl_interop,
            "_check_sass",
            False,
        )

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    h_input = np.array([1, 2, 3, 4, 5], dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, dtype)

    cuda.compute.inclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.PLUS,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.array([1, 3, 6, 10, 15])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


@pytest.mark.xfail(
    reason="CCCL_MAXIMUM well-known operation fails with NVRTC compilation error in C++ library"
)
def test_exclusive_scan_well_known_maximum():
    dtype = np.int32
    h_init = np.array([1], dtype=dtype)
    h_input = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, dtype)

    cuda.compute.exclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.MAXIMUM,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_scan_transform_output_iterator(floating_array):
    """Test scan with TransformOutputIterator."""
    dtype = floating_array.dtype
    h_init = np.array([0], dtype=dtype)

    # Use the floating_array fixture which provides random floating-point data of size 1000
    h_input = floating_array
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, dtype)

    def square(x: dtype) -> dtype:
        return x * x

    d_out_it = TransformOutputIterator(d_output, square)

    cuda.compute.inclusive_scan(
        d_in=d_input,
        d_out=d_out_it,
        op=OpKind.PLUS,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.cumsum(h_input) ** 2
    # Use more lenient tolerance for float32 due to precision differences
    if dtype == np.float32:
        np.testing.assert_allclose(
            d_output.copy_to_host(), expected, atol=1e-4, rtol=1e-4
        )
    else:
        np.testing.assert_allclose(d_output.copy_to_host(), expected, atol=1e-6)


def test_exclusive_scan_max():
    def max_op(a, b):
        return max(a, b)

    h_init = np.array([1], dtype="int32")
    h_input = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    cuda.compute.exclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=max_op,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.asarray([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_inclusive_scan_add():
    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    h_input = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    cuda.compute.inclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=add_op,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.asarray([-5, -5, -3, -6, -4, 0, 0, -1, 1, 9])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_reverse_input_iterator(monkeypatch):
    cc_major, _ = get_compute_capability()
    # Skip SASS check for CC 9.0+, due to a bug in NVRTC.
    # TODO: add NVRTC version check, ref nvbug 5243118
    if cc_major >= 9:
        import cuda.compute._cccl_interop as cccl_interop

        monkeypatch.setattr(
            cccl_interop,
            "_check_sass",
            False,
        )

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    h_input = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)
    reverse_it = ReverseIterator(d_input)

    cuda.compute.inclusive_scan(
        d_in=reverse_it,
        d_out=d_output,
        op=add_op,
        init_value=h_init,
        num_items=h_input.size,
    )

    # Check the result is correct
    expected = np.asarray([8, 10, 9, 9, 13, 15, 12, 14, 14, 9])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


@pytest.mark.no_verify_sass(reason="LDL/STL instructions emitted for this test.")
def test_reverse_output_iterator():
    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    h_input = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)
    reverse_it = ReverseIterator(d_output)

    cuda.compute.inclusive_scan(
        d_in=d_input,
        d_out=reverse_it,
        op=add_op,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.asarray([9, 1, -1, 0, 0, -4, -6, -3, -5, -5])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


@pytest.mark.parametrize(
    "force_inclusive",
    [True, False],
)
def test_future_init_value(force_inclusive):
    num_items = 1024
    dtype = np.dtype("int32")

    h_input = np.random.randint(0, 256, num_items, dtype=dtype)
    h_init = np.array([42], dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)
    init_value = DeviceArray.from_numpy(h_init)

    scan_device(d_input, d_output, num_items, OpKind.PLUS, init_value, force_inclusive)

    got = d_output.copy_to_host()
    expected = scan_host(h_input, lambda a, b: a + b, h_init, force_inclusive)
    np.testing.assert_array_equal(expected, got)


def test_no_init_value(monkeypatch):
    force_inclusive = True
    num_items = 1024
    dtype = np.dtype("int32")

    # Skip SASS check for CC 9.0 due to LDL/STL CI failure.
    cc_major, _ = get_compute_capability()
    if cc_major >= 9:
        import cuda.compute._cccl_interop

        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )

    h_input = np.random.randint(0, 256, num_items, dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    scan_device(d_input, d_output, num_items, OpKind.PLUS, None, force_inclusive)

    got = d_output.copy_to_host()
    expected = scan_host(h_input, lambda a, b: a + b, [0], force_inclusive)
    np.testing.assert_array_equal(expected, got)


def test_no_init_value_iterator():
    force_inclusive = True
    num_items = 1024
    dtype = np.dtype("float64")

    d_input = CountingIterator(np.float64(0))
    d_output = DeviceArray.empty(num_items, dtype)

    scan_device(d_input, d_output, num_items, OpKind.PLUS, None, force_inclusive)

    got = d_output.copy_to_host()
    expected = scan_host(
        np.arange(0, num_items, dtype=dtype), lambda a, b: a + b, [0], force_inclusive
    )

    np.testing.assert_array_equal(expected, got)


def test_inclusive_scan_with_lambda():
    """Test inclusive_scan with a lambda function as the scan operator."""
    h_init = np.array([0], dtype=np.int32)
    h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    # Use a lambda function directly as the scan operator
    cuda.compute.inclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=lambda a, b: a + b,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.array([1, 3, 6, 10, 15], dtype=np.int32)
    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


@pytest.mark.parametrize("force_inclusive", [True, False])
def test_scan_bool_maximum(force_inclusive):
    h_init = np.array([False], dtype=np.bool_)
    h_input = np.array([False, True, False, True], dtype=np.bool_)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    scan_device(
        d_input, d_output, h_input.size, OpKind.MAXIMUM, h_init, force_inclusive
    )

    if force_inclusive:
        expected = np.array([False, True, True, True], dtype=np.bool_)
    else:
        expected = np.array([False, False, True, True], dtype=np.bool_)

    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


def _run(scanner, *, d_in, d_out, op, init_value, num_items):
    bytes_needed = scanner(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        op=op,
        init_value=init_value,
        num_items=num_items,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    scanner(
        temp_storage=tmp,
        d_in=d_in,
        d_out=d_out,
        op=op,
        init_value=init_value,
        num_items=num_items,
    )


@pytest.mark.serialization
def test_serialize_deserialize_exclusive_scan_round_trip():
    h_in = np.arange(1, 33, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)
    init_value = np.array([0], dtype=np.int32)

    builder = make_exclusive_scan(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=init_value
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        init_value=init_value,
        num_items=h_in.size,
    )

    expected = np.zeros_like(h_in)
    np.cumsum(h_in[:-1], out=expected[1:])
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


@pytest.mark.serialization
def test_serialize_deserialize_inclusive_scan_round_trip():
    h_in = np.arange(1, 33, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)
    init_value = np.array([0], dtype=np.int32)

    builder = make_inclusive_scan(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=init_value
    )
    blob = serialize(builder)

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        init_value=init_value,
        num_items=h_in.size,
    )

    np.testing.assert_array_equal(d_out.copy_to_host(), np.cumsum(h_in))


@pytest.mark.serialization
def test_deserialize_after_jit_matches_jit_result():
    """Serialize a JITed scan, deserialize, and confirm output matches a fresh JIT."""
    h_in = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out_jit = DeviceArray.empty(h_in.shape, h_in.dtype)
    d_out_serialization = DeviceArray.empty(h_in.shape, h_in.dtype)
    init_value = np.array([1], dtype=np.int32)

    def max_op(a, b):
        return a if a > b else b

    # Build + serialize (this JITs), then clear every in-process cache so the serialization
    # leg below runs cold: the deserialized scan must stand on its own and cannot
    # free-ride on a callable warmed by the build or by the JIT reference.
    blob = serialize(
        make_exclusive_scan(
            d_in=d_in, d_out=d_out_serialization, op=max_op, init_value=init_value
        )
    )
    clear_all_caches()

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out_serialization,
        op=max_op,
        init_value=init_value,
        num_items=h_in.size,
    )

    # Compute the JIT reference only after the serialization path has already run.
    exclusive_scan(
        d_in=d_in,
        d_out=d_out_jit,
        op=max_op,
        init_value=init_value,
        num_items=h_in.size,
    )

    np.testing.assert_array_equal(
        d_out_serialization.copy_to_host(), d_out_jit.copy_to_host()
    )

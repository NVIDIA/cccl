# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import (
    ConstantIterator,
    CountingIterator,
    OpKind,
    TransformIterator,
    TransformOutputIterator,
    deserialize,
    gpu_struct,
    make_segmented_reduce,
    serialize,
)
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer


def is_out_of_memory_error(error):
    # cuda-core exception types vary by memory resource, so classify by message.
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "out_of_memory",
            "failed to allocate memory from pool",
        )
    )


@pytest.fixture(params=["i4", "u4", "i8", "u8"])
def offset_dtype(request):
    return np.dtype(request.param)


def test_segmented_reduce(input_array, offset_dtype, monkeypatch):
    "Test for all supported input types and for some offset types"
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )

    def binary_op(a, b):
        return a + b

    assert input_array.ndim == 1
    sz = input_array.size
    rng = np.random.default_rng()
    n_segments = 16
    h_offsets = np.zeros(n_segments + 1, dtype="int64")
    h_offsets[1:] = rng.multinomial(sz, [1 / n_segments] * n_segments)

    offsets = np.cumsum(np.asarray(h_offsets, dtype=offset_dtype), dtype=offset_dtype)

    h_start_offsets = offsets[:-1]
    h_end_offsets = offsets[1:]

    assert offsets.dtype == np.dtype(offset_dtype)
    assert np.all(h_start_offsets <= h_end_offsets)
    assert h_end_offsets[-1] == sz

    d_in = DeviceArray.from_numpy(input_array)
    d_out = DeviceArray.empty(n_segments, input_array.dtype)
    start_offsets = DeviceArray.from_numpy(h_start_offsets)
    end_offsets = DeviceArray.from_numpy(h_end_offsets)

    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    if input_array.dtype == np.float16:
        reduce_op = OpKind.PLUS
    else:
        reduce_op = binary_op

    # Call single-phase API directly with num_segments parameter
    cuda.compute.segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        num_segments=n_segments,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=reduce_op,
        h_init=h_init,
    )

    expected = np.empty(n_segments, dtype=input_array.dtype)
    for i in range(n_segments):
        expected[i] = np.sum(input_array[h_start_offsets[i] : h_end_offsets[i]])

    result = d_out.copy_to_host()
    if np.issubdtype(input_array.dtype, np.inexact):
        tolerance = 4 * np.finfo(input_array.dtype).eps
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
    else:
        np.testing.assert_array_equal(result, expected)


def test_segmented_reduce_struct_type(monkeypatch):
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )

    @gpu_struct
    class Pixel:
        r: np.int32
        g: np.int32
        b: np.int32

    def max_g_value(x, y):
        return x if x.g > y.g else y

    def align_up(n, m):
        return ((n + m - 1) // m) * m

    segment_size = 64
    n_pixels = align_up(4000, 64)
    offsets = np.arange(n_pixels + segment_size - 1, step=segment_size, dtype=np.int64)
    h_start_offsets = offsets[:-1]
    h_end_offsets = offsets[1:]
    n_segments = h_start_offsets.size

    rng = np.random.default_rng()
    h_rgb = rng.integers(0, 256, (n_pixels, 3), dtype=np.int32)
    h_rgb = h_rgb.view(Pixel.dtype).reshape(n_pixels)
    d_rgb = DeviceArray.from_numpy(h_rgb)
    d_out = DeviceArray.empty(n_segments, Pixel.dtype)
    start_offsets = DeviceArray.from_numpy(h_start_offsets)
    end_offsets = DeviceArray.from_numpy(h_end_offsets)

    h_init = Pixel(0, 0, 0)

    # Call single-phase API directly with n_segments parameter
    cuda.compute.segmented_reduce(
        d_in=d_rgb,
        d_out=d_out,
        num_segments=n_segments,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=max_g_value,
        h_init=h_init,
    )

    h_rgb = np.reshape(h_rgb, (n_segments, -1))
    expected = h_rgb[np.arange(h_rgb.shape[0]), h_rgb["g"].argmax(axis=-1)]

    np.testing.assert_equal(expected["g"], d_out.copy_to_host()["g"])


@pytest.mark.large
def test_large_num_segments_uniform_segment_sizes_nonuniform_input(monkeypatch):
    """
    This test verifies that segmented_reduce raises an error when
    num_segments exceeds 2^31-1.
    """
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )

    def make_difference(idx: np.int64) -> np.uint8:
        p = np.uint8(7)

        # Annotations on this nested function are intentionally omitted: on
        # Python 3.14 they cause numba_cuda to abort with `AssertionError:
        # unreachable` in op_SET_FUNCTION_ATTRIBUTE, because the new 0x10
        # (`__annotate__`, PEP 649) flag isn't handled.
        # Original signature: def Fu(idx: np.int64) -> np.uint8:
        def Fu(idx):
            i8 = np.uint8(idx % 5) + np.uint8(idx % 3)
            f = (i8 * (i8 + 1)) % p
            return f

        return (Fu(idx + 1) - Fu(idx)) % p

    input_it = TransformIterator(CountingIterator(np.int64(0)), make_difference)

    def make_scaler(step):
        def scale(row_id):
            return row_id * step

        return scale

    segment_size = 116
    offset0 = np.int64(0)
    row_offset = make_scaler(np.int64(segment_size))
    start_offsets = TransformIterator(CountingIterator(offset0), row_offset)
    end_offsets = start_offsets + 1

    num_segments = (2**15 + 2**3) * 2**16
    try:
        res = DeviceArray.empty(num_segments, np.uint8)
    except Exception as error:
        if not is_out_of_memory_error(error):
            raise
        pytest.skip("Insufficient memory to run the large number of segments test")
    assert res.nbytes == num_segments * np.dtype(np.uint8).itemsize

    def my_add(a: np.uint8, b: np.uint8) -> np.uint8:
        return (a + b) % np.uint8(7)

    h_init = np.zeros(tuple(), dtype=np.uint8)

    # Verify that the appropriate error is raised
    with pytest.raises(
        RuntimeError,
        match="Segmented sort does not currently support more than 2\\^31-1 segments\\.",
    ):
        cuda.compute.segmented_reduce(
            d_in=input_it,
            d_out=res,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
            op=my_add,
            h_init=h_init,
        )


@pytest.mark.large
def test_large_num_segments_nonuniform_segment_sizes_uniform_input(monkeypatch):
    """
    This test verifies that segmented_reduce raises an error when
    num_segments exceeds 2^31-1.
    """
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )
    input_it = ConstantIterator(np.int16(1))

    def offset_functor(m0: np.int64, p: np.int64):
        def offset_value(n: np.int64) -> np.int64:
            """
            Offset value computes closed form for
            :math:`sum(1 + (k % p), k=0..n)`.

            So segment lengths are periodic linearly
            increasing sequences, e.g,
            [min , min + 1, ..., min + p - 2,
                min + p - 1, min, min +1 , ....]
            """
            q = n // p
            r = n - q * p
            p2 = (p * (p - 1)) // 2
            r2 = (r * (r + 1)) // 2

            offset_val = (n + 1) * m0 + q * p2 + r2
            return offset_val

        return offset_value

    m0, p = np.int64(265), np.int64(163)
    offsets_it = TransformIterator(
        CountingIterator(np.int64(-1)), offset_functor(m0, p)
    )
    start_offsets = offsets_it
    end_offsets = offsets_it + 1

    def _plus(a, b):
        return a + b

    num_segments = (2**15 + 2**3) * 2**16
    try:
        res = DeviceArray.empty(num_segments, np.int16)
    except Exception as error:
        if not is_out_of_memory_error(error):
            raise
        pytest.skip("Insufficient memory to run the large number of segments test")
    assert res.nbytes == num_segments * np.dtype(np.int16).itemsize

    h_init = np.zeros(tuple(), dtype=np.int16)

    # Verify that the appropriate error is raised
    with pytest.raises(
        RuntimeError,
        match="Segmented sort does not currently support more than 2\\^31-1 segments\\.",
    ):
        cuda.compute.segmented_reduce(
            d_in=input_it,
            d_out=res,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
            op=_plus,
            h_init=h_init,
        )


def test_segmented_reduce_well_known_plus(monkeypatch):
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)

    # Create segmented data: [1, 2, 3] | [4, 5] | [6, 7, 8, 9]
    h_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
    h_starts = np.array([0, 3, 5], dtype=np.int32)
    h_ends = np.array([3, 5, 9], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_starts = DeviceArray.from_numpy(h_starts)
    d_ends = DeviceArray.from_numpy(h_ends)
    d_output = DeviceArray.empty(3, dtype)

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=3,
        start_offsets_in=d_starts,
        end_offsets_in=d_ends,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    expected = np.array([6, 9, 30])
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_segmented_reduce_well_known_maximum(monkeypatch):
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )
    dtype = np.int32
    h_init = np.array([-100], dtype=dtype)

    # Create segmented data: [1, 9, 3] | [4, 2] | [6, 7, 1, 8]
    h_input = np.array([1, 9, 3, 4, 2, 6, 7, 1, 8], dtype=dtype)
    h_starts = np.array([0, 3, 5], dtype=np.int32)
    h_ends = np.array([3, 5, 9], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_starts = DeviceArray.from_numpy(h_starts)
    d_ends = DeviceArray.from_numpy(h_ends)
    d_output = DeviceArray.empty(3, dtype)

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=3,
        start_offsets_in=d_starts,
        end_offsets_in=d_ends,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )

    expected = np.array([9, 4, 8])  # max of each segment
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_segmented_reduce_bool_maximum(monkeypatch):
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )
    h_init = np.array([False], dtype=np.bool_)

    # Create segmented data: [False, True] | [False, False] | [True]
    h_input = np.array([False, True, False, False, True], dtype=np.bool_)
    h_starts = np.array([0, 2, 4], dtype=np.int32)
    h_ends = np.array([2, 4, 5], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_starts = DeviceArray.from_numpy(h_starts)
    d_ends = DeviceArray.from_numpy(h_ends)
    d_output = DeviceArray.empty(3, np.bool_)

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=3,
        start_offsets_in=d_starts,
        end_offsets_in=d_ends,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )

    expected = np.array([True, False, True], dtype=np.bool_)
    np.testing.assert_equal(d_output.copy_to_host(), expected)


def test_segmented_reduce_transform_output_iterator(floating_array, monkeypatch):
    """Test segmented reduce with TransformOutputIterator."""
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )
    dtype = floating_array.dtype
    h_init = np.array([0], dtype=dtype)

    # Use the floating_array fixture which provides random floating-point data of size 1000
    d_input = DeviceArray.from_numpy(floating_array)

    # Create 2 segments of roughly equal size
    segment_size = floating_array.size // 2
    d_output = DeviceArray.empty(2, dtype)
    start_offsets = DeviceArray.from_numpy(np.array([0, segment_size], dtype=np.int32))
    end_offsets = DeviceArray.from_numpy(
        np.array([segment_size, floating_array.size], dtype=np.int32)
    )

    def sqrt(x: dtype) -> dtype:
        return x**0.5

    d_out_it = TransformOutputIterator(d_output, sqrt)

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_out_it,
        num_segments=2,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    expected = np.sqrt(
        np.array(
            [
                np.sum(floating_array[:segment_size]),
                np.sum(floating_array[segment_size:]),
            ]
        )
    )
    np.testing.assert_allclose(d_output.copy_to_host(), expected, atol=1e-6)


def test_device_segmented_reduce_for_rowwise_sum(monkeypatch):
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )

    def add_op(a, b):
        return a + b

    n_rows, n_cols = 67, 12345
    rng = np.random.default_rng()
    mat = rng.integers(low=-31, high=32, dtype=np.int32, size=(n_rows, n_cols))

    def make_scaler(step):
        def scale(row_id):
            return row_id * step

        return scale

    zero = np.int32(0)
    row_offset = make_scaler(np.int32(n_cols))
    start_offsets = TransformIterator(CountingIterator(zero), row_offset)

    end_offsets = start_offsets + 1

    d_input = DeviceArray.from_numpy(mat)
    h_init = np.zeros(tuple(), dtype=np.int32)
    d_output = DeviceArray.empty(n_rows, mat.dtype)

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=n_rows,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=add_op,
        h_init=h_init,
    )

    expected = np.sum(mat, axis=-1)
    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


def test_segmented_reduce_with_lambda(monkeypatch):
    """Test segmented_reduce with a lambda function as the reducer."""
    # Disable SASS verification for this test (LDL instruction in SASS).
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)

    # Create segmented data: [1, 2, 3] | [4, 5] | [6, 7, 8, 9]
    h_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
    h_starts = np.array([0, 3, 5], dtype=np.int32)
    h_ends = np.array([3, 5, 9], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_starts = DeviceArray.from_numpy(h_starts)
    d_ends = DeviceArray.from_numpy(h_ends)
    d_output = DeviceArray.empty(3, dtype)

    # Use a lambda function directly as the reducer
    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=3,
        start_offsets_in=d_starts,
        end_offsets_in=d_ends,
        op=lambda a, b: a + b,
        h_init=h_init,
    )

    expected = np.array([6, 9, 30])  # sum of each segment
    np.testing.assert_equal(d_output.copy_to_host(), expected)


@pytest.mark.parametrize(
    "max_seg_size",
    [
        4,  # small: warp-level reduction path
        64,  # medium: warp-level reduction path
        512,  # large: block-level reduction path
    ],
)
def test_segmented_reduce_max_segment_size(max_seg_size, monkeypatch):
    """Test that max_segment_size hint produces correct results with non-uniform segments.

    max_segment_size is a performance hint that selects an optimized kernel
    dispatch path. Segments vary in size from 1 to max_seg_size elements.
    """
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )
    dtype = np.int32
    rng = np.random.default_rng()
    num_segments = 1024
    h_init = np.zeros(1, dtype=dtype)

    # Non-uniform segment sizes in [1, max_seg_size]
    sizes = rng.integers(1, max_seg_size + 1, size=num_segments, dtype=np.int64)
    offsets = np.zeros(num_segments + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)

    total = int(offsets[-1])
    h_input = rng.integers(0, 100, size=total, dtype=dtype)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(num_segments, dtype)

    h_starts = offsets[:-1]
    h_ends = offsets[1:]
    d_starts = DeviceArray.from_numpy(h_starts)
    d_ends = DeviceArray.from_numpy(h_ends)

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=num_segments,
        start_offsets_in=d_starts,
        end_offsets_in=d_ends,
        op=OpKind.PLUS,
        h_init=h_init,
        max_segment_size=max_seg_size,
    )

    expected = np.empty(num_segments, dtype=dtype)
    for i in range(num_segments):
        expected[i] = np.sum(h_input[h_starts[i] : h_ends[i]])

    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


def _run(reducer, *, d_in, d_out, num_segments, start, end, op, h_init):
    bytes_needed = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_segments=num_segments,
        start_offsets_in=start,
        end_offsets_in=end,
        op=op,
        h_init=h_init,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    reducer(
        temp_storage=tmp,
        d_in=d_in,
        d_out=d_out,
        num_segments=num_segments,
        start_offsets_in=start,
        end_offsets_in=end,
        op=op,
        h_init=h_init,
    )


@pytest.mark.serialization
def test_serialize_deserialize_segmented_reduce_round_trip():
    h_in = np.array([8, 6, 7, 5, 3, 0, 9, -4, 3, 0, 1, 3, 1, 11, 25, 8], dtype=np.int32)
    offsets = np.array([0, 7, 11, 16], dtype=np.int64)
    d_in = DeviceArray.from_numpy(h_in)
    start = DeviceArray.from_numpy(offsets[:-1])
    end = DeviceArray.from_numpy(offsets[1:])
    n_segments = offsets.size - 1
    d_out = DeviceArray.empty(n_segments, np.int32)
    h_init = np.array([0], dtype=np.int32)

    builder = make_segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        start_offsets_in=start,
        end_offsets_in=end,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out,
        num_segments=n_segments,
        start=start,
        end=end,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    expected = np.array(
        [h_in[s:e].sum() for s, e in zip(offsets[:-1], offsets[1:])], dtype=np.int32
    )
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.cccl.parallel.experimental as parallel


@pytest.fixture(params=["i4", "u4", "i8", "u8"])
def offset_dtype(request):
    return np.dtype(request.param)


def test_segmented_reduce(input_array, offset_dtype):
    "Test for all supported input types and for some offset types"

    def binary_op(a, b):
        return a + b

    assert input_array.ndim == 1
    sz = input_array.size
    rng = cp.random
    n_segments = 16
    h_offsets = cp.zeros(n_segments + 1, dtype="int64")
    h_offsets[1:] = rng.multinomial(sz, [1 / n_segments] * n_segments)

    offsets = cp.cumsum(cp.asarray(h_offsets, dtype=offset_dtype), dtype=offset_dtype)

    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]

    assert offsets.dtype == np.dtype(offset_dtype)
    assert cp.all(start_offsets <= end_offsets)
    assert end_offsets[-1] == sz

    d_in = cp.asarray(input_array)
    d_out = cp.empty(n_segments, dtype=d_in.dtype)

    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    if input_array.dtype == np.float16:
        reduce_op = parallel.OpKind.PLUS
    else:
        reduce_op = binary_op

    # Call single-phase API directly with num_segments parameter
    parallel.segmented_reduce(
        d_in, d_out, start_offsets, end_offsets, reduce_op, h_init, n_segments
    )

    d_expected = cp.empty_like(d_out)
    for i in range(n_segments):
        d_expected[i] = cp.sum(d_in[start_offsets[i] : end_offsets[i]])

    assert cp.all(d_out == d_expected)


def test_segmented_reduce_struct_type():
    import cupy as cp
    import numpy as np

    @parallel.gpu_struct
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
    offsets = cp.arange(n_pixels + segment_size - 1, step=segment_size, dtype=np.int64)
    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]
    n_segments = start_offsets.size

    d_rgb = cp.random.randint(0, 256, (n_pixels, 3), dtype=np.int32).view(Pixel.dtype)
    d_out = cp.empty(n_segments, Pixel.dtype)

    h_init = Pixel(0, 0, 0)

    # Call single-phase API directly with n_segments parameter
    parallel.segmented_reduce(
        d_rgb, d_out, start_offsets, end_offsets, max_g_value, h_init, n_segments
    )

    h_rgb = np.reshape(d_rgb.get(), (n_segments, -1))
    expected = h_rgb[np.arange(h_rgb.shape[0]), h_rgb["g"].argmax(axis=-1)]

    np.testing.assert_equal(expected["g"], d_out.get()["g"])


@pytest.mark.large
def test_large_num_segments_uniform_segment_sizes_nonuniform_input():
    """
    This test builds input iterator as transformation
    over counting iterator by a function
    k -> (F(k + 1) - F(k)) % 7

    Segmented reduction with fixed size is performed
    using add modulo 7. Expected result is known to be
    F(end_offset[k] + 1) - F(start_offset[k]) % 7
    """

    def make_difference(idx: np.int64) -> np.uint8:
        p = np.uint8(7)

        def Fu(idx: np.int64) -> np.uint8:
            i8 = np.uint8(idx % 5) + np.uint8(idx % 3)
            f = (i8 * (i8 + 1)) % p
            return f

        return (Fu(idx + 1) - Fu(idx)) % p

    input_it = parallel.TransformIterator(
        parallel.CountingIterator(np.int64(0)), make_difference
    )

    def make_scaler(step):
        def scale(row_id):
            return row_id * step

        return scale

    segment_size = 116
    offset0 = np.int64(0)
    row_offset = make_scaler(np.int64(segment_size))
    start_offsets = parallel.TransformIterator(
        parallel.CountingIterator(offset0), row_offset
    )
    end_offsets = start_offsets + 1

    num_segments = (2**15 + 2**3) * 2**16
    try:
        res = cp.full(num_segments, fill_value=127, dtype=cp.uint8)
    except cp.cuda.memory.OutOfMemoryError:
        pytest.skip("Insufficient memory to run the large number of segments test")
    assert res.size == num_segments

    def my_add(a: np.uint8, b: np.uint8) -> np.uint8:
        return (a + b) % np.uint8(7)

    h_init = np.zeros(tuple(), dtype=np.uint8)
    # Call single-phase API directly with num_segments parameter
    parallel.segmented_reduce(
        input_it, res, start_offsets, end_offsets, my_add, h_init, num_segments
    )

    # Validation

    def get_expected_value(k: np.int64) -> np.uint8:
        i = np.uint8(k % 5) + np.uint8(k % 3)
        k1 = (k % 15) + (segment_size % 15)
        i1 = np.uint8(k1 % 5) + np.uint8(k1 % 3)
        p = np.uint8(7)
        v1 = np.uint8((i1 * (i1 + 1)) % p)
        v = np.uint8((i * (i + 1)) % p)
        return (v1 + (p - v)) % p

    # reset the iterator since it has been mutated by being incremented on host
    start_offsets.cvalue = type(start_offsets.cvalue)(offset0)
    expected = parallel.TransformIterator(start_offsets, get_expected_value)

    def cmp_op(a: np.uint8, b: np.uint8) -> np.uint8:
        return np.uint8(1) if (a == b) else np.uint8(0)

    validate = cp.zeros(2**20, dtype=np.uint8)

    id = 0
    while id < res.size:
        id_next = min(id + validate.size, res.size)
        num_items = id_next - id
        parallel.binary_transform(res[id:], expected + id, validate, cmp_op, num_items)
        assert id == (expected + id).cvalue.value
        assert cp.all(validate[:num_items].view(np.bool_))
        id = id_next


@pytest.mark.large
def test_large_num_segments_nonuniform_segment_sizes_uniform_input():
    """
    Test with large num_segments > INT_MAX

    Input is constant iterator with value 1.

    offset positions are computed as transformation
    over counting iterator with `n -> sum(min + (k % p), k=0..n)`.
    The closed form value of the sum is coded in `offset_value`
    function.

    Result of segmented reduction is known, and is
    given by transformed iterator over counting iterator
    transformed by `k -> min + (k % p)` function.
    """
    input_it = parallel.ConstantIterator(np.int16(1))

    def offset_functor(m0: np.int64, p: np.int64):
        def offset_value(n: np.int64):
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
    offsets_it = parallel.TransformIterator(
        parallel.CountingIterator(np.int64(-1)), offset_functor(m0, p)
    )
    start_offsets = offsets_it
    end_offsets = offsets_it + 1

    def _plus(a, b):
        return a + b

    num_segments = (2**15 + 2**3) * 2**16
    try:
        res = cp.full(num_segments, fill_value=-1, dtype=cp.int16)
    except cp.cuda.memory.OutOfMemoryError:
        pytest.skip("Insufficient memory to run the large number of segments test")
    assert res.size == num_segments

    h_init = np.zeros(tuple(), dtype=np.int16)
    # Call single-phase API directly with num_segments parameter
    parallel.segmented_reduce(
        input_it, res, start_offsets, end_offsets, _plus, h_init, num_segments
    )

    # Validation

    def get_expected_value(k: np.int64) -> np.int16:
        return np.int16(m0 + (k % p))

    expected = parallel.TransformIterator(
        parallel.CountingIterator(np.int64(0)), get_expected_value
    )

    def cmp_op(a: np.int16, b: np.int16) -> np.uint8:
        return np.uint8(1) if (a == b) else np.uint8(0)

    validate = cp.zeros(2**20, dtype=np.uint8)

    id = 0
    while id < res.size:
        id_next = min(id + validate.size, res.size)
        num_items = id_next - id
        parallel.binary_transform(res[id:], expected + id, validate, cmp_op, num_items)
        assert id == (expected + id).cvalue.value
        assert cp.all(validate[:num_items].view(np.bool_))
        id = id_next


def test_segmented_reduce_well_known_plus():
    """Test segmented reduce with well-known PLUS operation."""
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)

    # Create segmented data: [1, 2, 3] | [4, 5] | [6, 7, 8, 9]
    d_input = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
    d_starts = cp.array([0, 3, 5], dtype=np.int32)
    d_ends = cp.array([3, 5, 9], dtype=np.int32)
    d_output = cp.empty(3, dtype=dtype)

    # Run segmented reduce with well-known PLUS operation
    parallel.segmented_reduce(
        d_input, d_output, d_starts, d_ends, parallel.OpKind.PLUS, h_init, 3
    )

    # Check the result is correct
    expected = np.array([6, 9, 30])  # sums of each segment
    np.testing.assert_equal(d_output.get(), expected)


@pytest.mark.xfail(reason="MAXIMUM op is not implemented. See GH #5515")
def test_segmented_reduce_well_known_maximum():
    """Test segmented reduce with well-known MAXIMUM operation."""
    dtype = np.int32
    h_init = np.array([-100], dtype=dtype)

    # Create segmented data: [1, 9, 3] | [4, 2] | [6, 7, 1, 8]
    d_input = cp.array([1, 9, 3, 4, 2, 6, 7, 1, 8], dtype=dtype)
    d_starts = cp.array([0, 3, 5], dtype=np.int32)
    d_ends = cp.array([3, 5, 9], dtype=np.int32)
    d_output = cp.empty(3, dtype=dtype)

    # Run segmented reduce with well-known MAXIMUM operation
    parallel.segmented_reduce(
        d_input, d_output, d_starts, d_ends, parallel.OpKind.MAXIMUM, h_init, 3
    )

    # Check the result is correct
    expected = np.array([9, 4, 8])  # max of each segment
    np.testing.assert_equal(d_output.get(), expected)

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.parallel.experimental.algorithms.legacy as algorithms
from cuda.parallel.experimental.struct import gpu_struct


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

    segmented_reduce_fn = algorithms.segmented_reduce(
        d_in, d_out, start_offsets, end_offsets, binary_op, h_init
    )

    temp_nbytes = segmented_reduce_fn(
        None, d_in, d_out, n_segments, start_offsets, end_offsets, h_init
    )
    temp = cp.empty(temp_nbytes, dtype="uint8")

    segmented_reduce_fn(
        temp, d_in, d_out, n_segments, start_offsets, end_offsets, h_init
    )

    d_expected = cp.empty_like(d_out)
    for i in range(n_segments):
        d_expected[i] = cp.sum(d_in[start_offsets[i] : end_offsets[i]])

    assert cp.all(d_out == d_expected)


def test_segmented_reduce_struct_type():
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms.legacy as algorithms

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
    offsets = cp.arange(n_pixels + segment_size - 1, step=segment_size, dtype=np.int64)
    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]
    n_segments = start_offsets.size

    d_rgb = cp.random.randint(0, 256, (n_pixels, 3), dtype=np.int32).view(Pixel.dtype)
    d_out = cp.empty(n_segments, Pixel.dtype)

    h_init = Pixel(0, 0, 0)

    alg = algorithms.segmented_reduce(
        d_rgb, d_out, start_offsets, end_offsets, max_g_value, h_init
    )
    temp_storage_bytes = alg(
        None, d_rgb, d_out, n_segments, start_offsets, end_offsets, h_init
    )

    d_temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
    _ = alg(
        d_temp_storage, d_rgb, d_out, n_segments, start_offsets, end_offsets, h_init
    )

    h_rgb = np.reshape(d_rgb.get(), (n_segments, -1))
    expected = h_rgb[np.arange(h_rgb.shape[0]), h_rgb["g"].argmax(axis=-1)]

    np.testing.assert_equal(expected["g"], d_out.get()["g"])

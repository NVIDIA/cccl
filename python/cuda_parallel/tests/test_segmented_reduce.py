# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms


def test_segmented_reduce(input_array):
    def binary_op(a, b):
        return a + b

    assert input_array.ndim == 1
    sz = input_array.size
    rng = np.random.default_rng()
    n_segments = 2**4
    h_offsets = np.zeros(n_segments + 1, dtype="int64")
    h_offsets[1:] = rng.multinomial(sz, [1 / 16] * 16)

    offsets = cp.asarray(h_offsets)

    start_offsets = offsets[:-1]
    end_offsets = offsets[:-1]

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

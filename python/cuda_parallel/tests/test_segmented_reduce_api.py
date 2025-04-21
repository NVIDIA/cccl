# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_device_segmented_reduce():
    # example-begin segmented-reduce-min
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms

    def min_op(a, b):
        return a if a < b else b

    dtype = np.dtype(np.int32)
    max_val = np.iinfo(dtype).max
    h_init = np.asarray(max_val, dtype=dtype)

    offsets = cp.array([0, 7, 11, 16], dtype=np.int64)
    first_segment = (8, 6, 7, 5, 3, 0, 9)
    second_segment = (-4, 3, 0, 1)
    third_segment = (3, 1, 11, 25, 8)
    d_input = cp.array(
        [*first_segment, *second_segment, *third_segment],
        dtype=dtype,
    )

    start_o = offsets[:-1]
    end_o = offsets[1:]

    n_segments = start_o.size
    d_output = cp.empty(n_segments, dtype=dtype)

    # Instantiate reduction for the given operator and initial value
    segmented_reduce = algorithms.segmented_reduce(
        d_output, d_output, start_o, end_o, min_op, h_init
    )

    # Determine temporary device storage requirements
    temp_storage_size = segmented_reduce(
        None, d_input, d_output, n_segments, start_o, end_o, h_init
    )

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    segmented_reduce(
        d_temp_storage, d_input, d_output, n_segments, start_o, end_o, h_init
    )

    # Check the result is correct
    expected_output = cp.asarray([0, -4, 1], dtype=d_output.dtype)
    assert (d_output == expected_output).all()
    # example-end segmented-reduce-min


def test_device_segmented_reduce_for_rowwise_sum():
    # example-begin segmented-reduce-rowwise-sum
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms
    import cuda.parallel.experimental.iterators as iterators

    def add_op(a, b):
        return a + b

    n_rows, n_cols = 67, 12345
    rng = cp.random.default_rng()
    mat = rng.integers(low=-31, high=32, dtype=np.int32, size=(n_rows, n_cols))

    def make_scaler(step):
        def scale(row_id):
            return row_id * step

        return scale

    zero = np.int32(0)
    row_offset = make_scaler(np.int32(n_cols))
    start_offsets = iterators.TransformIterator(
        iterators.CountingIterator(zero), row_offset
    )

    end_offsets = start_offsets + 1

    d_input = mat
    h_init = np.zeros(tuple(), dtype=np.int32)
    d_output = cp.empty(n_rows, dtype=d_input.dtype)

    alg = algorithms.segmented_reduce(
        d_input, d_output, start_offsets, end_offsets, add_op, h_init
    )

    # query size of temporary storage and allocate
    temp_nbytes = alg(
        None, d_input, d_output, n_rows, start_offsets, end_offsets, h_init
    )
    temp_storage = cp.empty(temp_nbytes, dtype=cp.uint8)
    # launch computation
    alg(temp_storage, d_input, d_output, n_rows, start_offsets, end_offsets, h_init)

    # Verify correctness
    expected = cp.sum(mat, axis=-1)
    assert cp.all(d_output == expected)
    # example-end segmented-reduce-columnwise-total

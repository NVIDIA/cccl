# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Segmented reduction examples demonstrating reduction operations on segmented data.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def basic_segmented_reduce_example():
    """Demonstrate basic segmented reduction finding minimum in each segment."""

    def min_op(a, b):
        return a if a < b else b

    dtype = np.dtype(np.int32)
    max_val = np.iinfo(dtype).max
    h_init = np.asarray(max_val, dtype=dtype)

    # Define segments: [8,6,7,5,3,0,9], [-4,3,0,1], [3,1,11,25,8]
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

    # Run segmented reduction with automatic temp storage allocation
    parallel.segmented_reduce(
        d_input, d_output, start_o, end_o, min_op, h_init, n_segments
    )

    # Check the result is correct
    expected_output = cp.asarray([0, -4, 1], dtype=d_output.dtype)
    assert (d_output == expected_output).all()
    print(f"Segmented min results: {d_output.get()}")
    return d_output.get()


def rowwise_sum_example():
    """Demonstrate segmented reduction for computing row-wise sums of a matrix."""
    n_rows, n_cols = 5, 4  # Smaller example for clarity
    # Create a simple matrix for demonstration
    mat = cp.array(
        [
            [1, 2, 3, 4],  # Row 0: sum = 10
            [5, 6, 7, 8],  # Row 1: sum = 26
            [9, 10, 11, 12],  # Row 2: sum = 42
            [13, 14, 15, 16],  # Row 3: sum = 58
            [17, 18, 19, 20],  # Row 4: sum = 74
        ],
        dtype=np.int32,
    )

    def make_scaler(step):
        def scale(row_id):
            return row_id * step

        return scale

    zero = np.int32(0)
    row_offset = make_scaler(np.int32(n_cols))
    start_offsets = parallel.TransformIterator(
        parallel.CountingIterator(zero), row_offset
    )

    end_offsets = start_offsets + 1

    d_input = mat
    h_init = np.zeros(tuple(), dtype=np.int32)
    d_output = cp.empty(n_rows, dtype=d_input.dtype)

    # Run segmented reduction with automatic temp storage allocation
    parallel.segmented_reduce(
        d_input,
        d_output,
        start_offsets,
        end_offsets,
        parallel.OpKind.PLUS,
        h_init,
        n_rows,
    )

    # Verify correctness
    expected = cp.sum(mat, axis=-1)
    assert cp.all(d_output == expected)
    print(f"Row-wise sums: {d_output.get()}")
    print(f"Expected: {expected.get()}")
    return d_output.get()


def mixed_segments_example():
    """Demonstrate segmented reduction with different segment sizes."""

    # Create segments of different sizes
    # Segment 0: [1] (size 1)
    # Segment 1: [2, 3, 4] (size 3)
    # Segment 2: [5, 6] (size 2)
    # Segment 3: [7, 8, 9, 10] (size 4)
    offsets = cp.array([0, 1, 4, 6, 10], dtype=np.int64)
    d_input = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)

    start_o = offsets[:-1]
    end_o = offsets[1:]

    n_segments = start_o.size
    d_output = cp.empty(n_segments, dtype=np.int32)
    h_init = np.array(0, dtype=np.int32)

    # Run segmented reduction with automatic temp storage allocation
    parallel.segmented_reduce(
        d_input, d_output, start_o, end_o, parallel.OpKind.PLUS, h_init, n_segments
    )

    # Expected results: [1, 9, 11, 34]
    expected_output = cp.asarray([1, 9, 11, 34], dtype=d_output.dtype)
    assert (d_output == expected_output).all()
    print(f"Mixed segments sums: {d_output.get()}")
    return d_output.get()


if __name__ == "__main__":
    print("Running segmented reduce examples...")
    basic_segmented_reduce_example()
    rowwise_sum_example()
    mixed_segments_example()
    print("All segmented reduce examples completed successfully!")

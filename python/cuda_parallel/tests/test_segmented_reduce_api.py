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
    d_input = cp.array(
        [
            8,
            6,
            7,
            5,
            3,
            0,
            9,  # first segment
            -4,
            3,
            0,
            1,  # second segment
            3,
            1,
            11,
            25,
            8,  # third segment
        ],
        dtype=dtype,
    )
    d_output = cp.empty(offsets.size - 1, dtype=dtype)

    # Instantiate reduction for the given operator and initial value
    segmented_reduce = algorithms.segmented_reduce(
        d_output, d_output, offsets, offsets, min_op, h_init
    )

    # Determine temporary device storage requirements
    temp_storage_size = segmented_reduce(
        None, d_input, d_output, d_output.size, offsets[:-1], offsets[1:], h_init
    )

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    segmented_reduce(
        d_temp_storage,
        d_input,
        d_output,
        d_output.size,
        offsets[:-1],
        offsets[1:],
        h_init,
    )

    # Check the result is correct
    expected_output = cp.asarray([0, -4, 1], dtype=d_output.dtype)
    assert (d_output == expected_output).all()
    # example-end segmented-reduce-min

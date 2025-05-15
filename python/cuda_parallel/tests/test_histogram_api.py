# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_histogram_even():
    # example-begin histogram-even
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms

    num_samples = 10
    h_samples = np.array(
        [2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5], dtype="float32"
    )
    d_samples = cp.asarray(h_samples)
    num_levels = 7
    h_num_output_levels = np.array([num_levels], dtype="int32")
    d_histogram = cp.empty(num_levels - 1, dtype="int32")
    h_lower_level = np.array([0], dtype="float64")
    h_upper_level = np.array([12], dtype="float64")

    # Instantiate histogram for the given samples, output histogram, number of levels, and levels
    histogram = algorithms.histogram(
        d_samples, d_histogram, h_num_output_levels, h_lower_level, num_samples
    )

    # Determine temporary device storage requirements
    temp_storage_size = histogram(
        None,
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        num_samples,
    )

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run histogram
    histogram(
        d_temp_storage,
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        num_samples,
    )

    h_actual_histogram = cp.asnumpy(d_histogram)
    h_expected_histogram = np.array([1, 5, 0, 3, 0, 0], dtype="int32")

    np.testing.assert_array_equal(h_actual_histogram, h_expected_histogram)
    # example-end histogram-even

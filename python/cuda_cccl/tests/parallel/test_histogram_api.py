# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_histogram_even():
    # example-begin histogram-even
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental as parallel

    num_samples = 10
    h_samples = np.array(
        [2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5], dtype="float32"
    )
    d_samples = cp.asarray(h_samples)
    num_levels = 7
    d_histogram = cp.empty(num_levels - 1, dtype="int32")
    lower_level = np.float64(0)
    upper_level = np.float64(12)

    # Run histogram with automatic temp storage allocation
    parallel.histogram_even(
        d_samples,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        num_samples,
    )

    # Check the result is correct
    h_actual_histogram = cp.asnumpy(d_histogram)
    # Calculate expected histogram using numpy
    h_expected_histogram, _ = np.histogram(
        h_samples, bins=num_levels - 1, range=(lower_level, upper_level)
    )
    h_expected_histogram = h_expected_histogram.astype("int32")

    np.testing.assert_array_equal(h_actual_histogram, h_expected_histogram)
    # example-end histogram-even

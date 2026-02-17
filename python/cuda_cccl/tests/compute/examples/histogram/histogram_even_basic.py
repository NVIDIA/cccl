# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use histogram_even to bin a sequence of samples.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and output arrays.
num_samples = 10
h_samples = np.array(
    [2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5], dtype="float32"
)
d_samples = cp.asarray(h_samples)
num_levels = 7
d_histogram = cp.empty(num_levels - 1, dtype="int32")
lower_level = np.float32(0)
upper_level = np.float32(12)

# Perform the histogram even.
cuda.compute.histogram_even(
    d_samples,
    d_histogram,
    num_levels,
    lower_level,
    upper_level,
    num_samples,
)

# Verify the result.
h_actual_histogram = cp.asnumpy(d_histogram)
h_expected_histogram, _ = np.histogram(
    h_samples, bins=num_levels - 1, range=(lower_level, upper_level)
)
h_expected_histogram = h_expected_histogram.astype("int32")

np.testing.assert_array_equal(h_actual_histogram, h_expected_histogram)
print(f"Histogram even basic result: {h_actual_histogram}")

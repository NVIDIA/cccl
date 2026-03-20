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

h_num_output_levels = np.array([num_levels], dtype=np.int32)
h_lower_level = np.array([lower_level], dtype=np.float32)
h_upper_level = np.array([upper_level], dtype=np.float32)

histogrammer = cuda.compute.make_histogram_even(
    d_samples,
    d_histogram,
    h_num_output_levels,
    h_lower_level,
    h_upper_level,
    num_samples,
)

temp_storage_bytes = int(
    histogrammer.get_temp_storage_bytes(
        d_samples,
        d_histogram,
        num_samples,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
    )
)

d_temp_storage = (
    None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
)

histogrammer.compute(
    d_temp_storage,
    d_samples,
    d_histogram,
    num_samples,
    h_num_output_levels=h_num_output_levels,
    h_lower_level=h_lower_level,
    h_upper_level=h_upper_level,
)

# Verify the result.
h_actual_histogram = cp.asnumpy(d_histogram)
h_expected_histogram, _ = np.histogram(
    h_samples, bins=num_levels - 1, range=(lower_level, upper_level)
)
h_expected_histogram = h_expected_histogram.astype("int32")

np.testing.assert_array_equal(h_actual_histogram, h_expected_histogram)
print(f"Histogram even basic result: {h_actual_histogram}")

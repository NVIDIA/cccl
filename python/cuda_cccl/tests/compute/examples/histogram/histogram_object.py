# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use histogram object API to bin a sequence of samples.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and output arrays.
h_samples = np.array(
    [1.5, 2.3, 4.7, 6.2, 7.8, 3.1, 5.5, 8.9, 2.7, 6.4], dtype="float32"
)
d_samples = cp.asarray(h_samples)

num_levels = 6

# note that the object API requires passing numpy arrays
# rather than scalars:
h_num_output_levels = np.array([num_levels], dtype=np.int32)
h_lower_level = np.array([0.0], dtype=np.float32)
h_upper_level = np.array([10.0], dtype=np.float32)

d_histogram = cp.zeros(num_levels - 1, dtype="int32")

# Create the histogram object.
histogrammer = cuda.compute.make_histogram_even(
    d_samples,
    d_histogram,
    h_num_output_levels,
    h_lower_level,
    h_upper_level,
    len(h_samples),
)

# Get the temporary storage size.
temp_storage_size = histogrammer(
    None,
    d_samples,
    d_histogram,
    h_num_output_levels,
    h_lower_level,
    h_upper_level,
    len(h_samples),
)

# Allocate the temporary storage.
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Perform the histogram.
histogrammer(
    d_temp_storage,
    d_samples,
    d_histogram,
    h_num_output_levels,
    h_lower_level,
    h_upper_level,
    len(h_samples),
)

# Verify the result.
h_result = cp.asnumpy(d_histogram)
expected_histogram = np.array([1, 3, 2, 3, 1], dtype="int32")

np.testing.assert_array_equal(h_result, expected_histogram)
print("Histogram object example completed successfully")

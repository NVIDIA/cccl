# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use segmented_sort with the object API.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and segment offsets.
dtype = np.int32
h_input_keys = np.array([9, 1, 5, 4, 2, 8, 7, 3, 6], dtype=dtype)
h_input_vals = np.array([90, 10, 50, 40, 20, 80, 70, 30, 60], dtype=dtype)
start_offsets = np.array([0, 3, 5], dtype=np.int64)
end_offsets = np.array([3, 5, 9], dtype=np.int64)

d_input_keys = cp.asarray(h_input_keys)
d_input_vals = cp.asarray(h_input_vals)
d_output_keys = cp.empty_like(d_input_keys)
d_output_vals = cp.empty_like(d_input_vals)

# Create the segmented sort object.
sorter = cuda.compute.make_segmented_sort(
    d_input_keys,
    d_output_keys,
    d_input_vals,
    d_output_vals,
    cp.asarray(start_offsets),
    cp.asarray(end_offsets),
    cuda.compute.SortOrder.ASCENDING,
)

# Get the temporary storage size.
temp_storage_size = sorter(
    None,
    d_input_keys,
    d_output_keys,
    d_input_vals,
    d_output_vals,
    h_input_keys.size,
    start_offsets.size,
    cp.asarray(start_offsets),
    cp.asarray(end_offsets),
)
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Perform the segmented sort.
sorter(
    d_temp_storage,
    d_input_keys,
    d_output_keys,
    d_input_vals,
    d_output_vals,
    h_input_keys.size,
    start_offsets.size,
    cp.asarray(start_offsets),
    cp.asarray(end_offsets),
)

# Verify the result.
expected_pairs = []
for s, e in zip(start_offsets, end_offsets):
    seg_pairs = sorted(zip(h_input_keys[s:e], h_input_vals[s:e]), key=lambda kv: kv[0])
    expected_pairs.extend(seg_pairs)

expected_keys = np.array([k for k, _ in expected_pairs], dtype=dtype)
expected_values = np.array([v for _, v in expected_pairs], dtype=dtype)

actual_keys = d_output_keys.get()
actual_values = d_output_vals.get()
np.testing.assert_array_equal(actual_keys, expected_keys)
np.testing.assert_array_equal(actual_values, expected_values)
print("Segmented sort object example completed successfully")

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use segmented_sort with DoubleBuffer for reduced temporary storage.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare input keys and values, and segment offsets.
h_in_keys = np.array([9, 1, 5, 4, 2, 8, 7, 3, 6], dtype="int32")
h_in_vals = np.array([90, 10, 50, 40, 20, 80, 70, 30, 60], dtype="int32")

# 3 segments: [0,3), [3,5), [5,9)
start_offsets = np.array([0, 3, 5], dtype=np.int64)
end_offsets = np.array([3, 5, 9], dtype=np.int64)

d_in_keys = cp.asarray(h_in_keys)
d_in_vals = cp.asarray(h_in_vals)
d_tmp_keys = cp.empty_like(d_in_keys)
d_tmp_vals = cp.empty_like(d_in_vals)

# Create double buffers for keys and values.
keys_db = cuda.compute.DoubleBuffer(d_in_keys, d_tmp_keys)
vals_db = cuda.compute.DoubleBuffer(d_in_vals, d_tmp_vals)

# Perform the segmented sort (descending within each segment).
cuda.compute.segmented_sort(
    keys_db,
    None,
    vals_db,
    None,
    d_in_keys.size,
    start_offsets.size,
    cp.asarray(start_offsets),
    cp.asarray(end_offsets),
    cuda.compute.SortOrder.DESCENDING,
)

# Verify the result.
h_out_keys = cp.asnumpy(keys_db.current())
h_out_vals = cp.asnumpy(vals_db.current())

expected_pairs = []
for s, e in zip(start_offsets, end_offsets):
    seg_pairs = sorted(
        zip(h_in_keys[s:e], h_in_vals[s:e]), key=lambda kv: kv[0], reverse=True
    )
    expected_pairs.extend(seg_pairs)

expected_keys = np.array([k for k, _ in expected_pairs], dtype=h_in_keys.dtype)
expected_vals = np.array([v for _, v in expected_pairs], dtype=h_in_vals.dtype)

assert np.array_equal(h_out_keys, expected_keys)
assert np.array_equal(h_out_vals, expected_vals)
print(f"Segmented sort buffer result - keys: {h_out_keys}, values: {h_out_vals}")

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Demonstrate basic merge sort with keys and values.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input and output arrays.
h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
h_in_values = np.array(
    [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
)

d_in_keys = cp.asarray(h_in_keys)
d_in_values = cp.asarray(h_in_values)

# Perform the merge sort.
parallel.merge_sort(
    d_in_keys,
    d_in_values,
    d_in_keys,
    d_in_values,
    parallel.OpKind.LESS,
    d_in_keys.size,
)

# Verify the result.
h_out_keys = cp.asnumpy(d_in_keys)
h_out_values = cp.asnumpy(d_in_values)

argsort = np.argsort(h_in_keys, stable=True)
expected_keys = np.array(h_in_keys)[argsort]
expected_values = np.array(h_in_values)[argsort]

assert np.array_equal(h_out_keys, expected_keys)
assert np.array_equal(h_out_values, expected_values)
print(f"Merge sort basic result - keys: {h_out_keys}, values: {h_out_values}")

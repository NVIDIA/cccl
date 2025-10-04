# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use unique_by_key to remove all
but the first value for each group of consecutive keys.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
)

# Prepare the input and output arrays.
h_in_keys = np.array([0, 2, 2, 9, 5, 5, 5, 8], dtype="int32")
h_in_values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="float32")

d_in_keys = cp.asarray(h_in_keys)
d_in_values = cp.asarray(h_in_values)
d_out_keys = cp.empty_like(d_in_keys)
d_out_values = cp.empty_like(d_in_values)
d_out_num_selected = cp.empty(1, np.int32)

# Perform the unique by key operation.
cuda.compute.unique_by_key(
    d_in_keys,
    d_in_values,
    d_out_keys,
    d_out_values,
    d_out_num_selected,
    OpKind.EQUAL_TO,
    d_in_keys.size,
)

# Verify the result.
num_selected = cp.asnumpy(d_out_num_selected)[0]
h_out_keys = cp.asnumpy(d_out_keys)[:num_selected]
h_out_values = cp.asnumpy(d_out_values)[:num_selected]

expected_keys = np.array([0, 2, 9, 5, 8])
expected_values = np.array([1, 2, 4, 5, 8])

assert np.array_equal(h_out_keys, expected_keys)
assert np.array_equal(h_out_values, expected_values)
print(
    f"Unique by key basic result - keys: {h_out_keys}, values: {h_out_values}, count: {num_selected}"
)

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use unique_by_key with the object API.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
)

# Unique by key example demonstrating the object API
dtype = np.int32
h_input_keys = np.array([1, 1, 2, 3, 3], dtype=dtype)
h_input_values = np.array([10, 20, 30, 40, 50], dtype=dtype)
d_input_keys = cp.asarray(h_input_keys)
d_input_values = cp.asarray(h_input_values)
d_output_keys = cp.empty_like(d_input_keys)
d_output_values = cp.empty_like(d_input_values)
d_num_selected = cp.empty(1, dtype=np.int32)

# Create the unique by key object.
uniquer = cuda.compute.make_unique_by_key(
    d_input_keys,
    d_input_values,
    d_output_keys,
    d_output_values,
    d_num_selected,
    OpKind.EQUAL_TO,
)

# Get the temporary storage size.
temp_storage_size = uniquer(
    None,
    d_input_keys,
    d_input_values,
    d_output_keys,
    d_output_values,
    d_num_selected,
    OpKind.EQUAL_TO,
    len(h_input_keys),
)

# Allocate the temporary storage.
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Perform the unique by key operation.
uniquer(
    d_temp_storage,
    d_input_keys,
    d_input_values,
    d_output_keys,
    d_output_values,
    d_num_selected,
    OpKind.EQUAL_TO,
    len(h_input_keys),
)

# Verify the result.
num_selected = d_num_selected.get()[0]
expected_keys = np.array([1, 2, 3], dtype=dtype)
expected_values = np.array([10, 30, 40], dtype=dtype)
actual_keys = d_output_keys.get()[:num_selected]
actual_values = d_output_values.get()[:num_selected]
np.testing.assert_array_equal(actual_keys, expected_keys)
np.testing.assert_array_equal(actual_values, expected_values)
print("Unique by key object example completed successfully")

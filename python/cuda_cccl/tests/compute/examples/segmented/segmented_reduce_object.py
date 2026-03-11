# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Segmented reduction using the object API.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
)

# Prepare the input and output arrays.
dtype = np.int32
h_init = np.array([0], dtype=dtype)
h_input = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
d_input = cp.asarray(h_input)
d_output = cp.empty(2, dtype=dtype)

start_offsets = cp.array([0, 3], dtype=np.int64)
end_offsets = cp.array([3, 6], dtype=np.int64)

# Create the segmented reduce object.
reducer = cuda.compute.make_segmented_reduce(
    d_input, d_output, start_offsets, end_offsets, OpKind.PLUS, h_init
)

# Get the temporary storage size.
temp_storage_size = reducer(
    None, d_input, d_output, OpKind.PLUS, 2, start_offsets, end_offsets, h_init
)

# Allocate the temporary storage.
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Perform the segmented reduce.
reducer(
    d_temp_storage,
    d_input,
    d_output,
    OpKind.PLUS,
    2,
    start_offsets,
    end_offsets,
    h_init,
)

# Verify the result.
expected_result = np.array([6, 15], dtype=dtype)
actual_result = d_output.get()
np.testing.assert_array_equal(actual_result, expected_result)
print("Segmented reduce object example completed successfully")

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Unary transform examples demonstrating the object API and well-known operations.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and output arrays.
dtype = np.int32
h_input = np.array([1, 2, 3, 4], dtype=dtype)
d_input = cp.asarray(h_input)
d_output = cp.empty_like(d_input)


# Define the unary operation.
def add_one_op(a):
    return a + 1


# Create the unary transform object.
transformer = cuda.compute.make_unary_transform(d_input, d_output, add_one_op)

# Perform the unary transform.
transformer(d_input, d_output, add_one_op, len(h_input))

# Verify the result.
expected_result = np.array([2, 3, 4, 5], dtype=dtype)
actual_result = d_output.get()
np.testing.assert_array_equal(actual_result, expected_result)
print("Unary transform object example completed successfully")

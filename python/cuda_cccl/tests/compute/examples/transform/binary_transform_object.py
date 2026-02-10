# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Binary transform examples demonstrating the transform object API.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
)

# Prepare the input and output arrays.
dtype = np.int32
h_input1 = np.array([1, 2, 3, 4], dtype=dtype)
h_input2 = np.array([10, 20, 30, 40], dtype=dtype)
d_input1 = cp.asarray(h_input1)
d_input2 = cp.asarray(h_input2)
d_output = cp.empty_like(d_input1)

# Create the binary transform object.
transformer = cuda.compute.make_binary_transform(
    d_input1, d_input2, d_output, OpKind.PLUS
)

# Perform the binary transform.
transformer(d_input1, d_input2, d_output, OpKind.PLUS, len(h_input1))

# Verify the result.
expected_result = np.array([11, 22, 33, 44], dtype=dtype)
actual_result = d_output.get()
np.testing.assert_array_equal(actual_result, expected_result)
print("Binary transform object example completed successfully")

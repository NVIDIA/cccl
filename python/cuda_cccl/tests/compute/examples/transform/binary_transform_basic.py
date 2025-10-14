# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use binary_transform to perform elementwise addition.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
)

# Prepare the input and output arrays.
input1_data = np.array([1, 2, 3, 4], dtype=np.int32)
input2_data = np.array([10, 20, 30, 40], dtype=np.int32)
d_in1 = cp.asarray(input1_data)
d_in2 = cp.asarray(input2_data)
d_out = cp.empty_like(d_in1)

# Perform the binary transform.
cuda.compute.binary_transform(d_in1, d_in2, d_out, OpKind.PLUS, len(d_in1))

# Verify the result.
result = d_out.get()
expected = input1_data + input2_data

np.testing.assert_array_equal(result, expected)
print(f"Binary transform result: {result}")

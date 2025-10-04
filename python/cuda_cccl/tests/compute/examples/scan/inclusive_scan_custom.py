# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Inclusive scan with custom operation (prefix sum of even values).
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and output arrays.
h_init = np.array([0], dtype="int32")
d_input = cp.array([1, 2, 3, 4, 5], dtype="int32")
d_output = cp.empty_like(d_input, dtype="int32")

# Define the binary operation for the scan.


def add_op(a, b):
    return (a if a % 2 == 0 else 0) + (b if b % 2 == 0 else 0)


# Perform the inclusive scan.
cuda.compute.inclusive_scan(d_input, d_output, add_op, h_init, d_input.size)

# Verify the result.
expected = np.asarray([0, 2, 2, 6, 6])
assert np.array_equal(d_output.get(), expected)
result = d_output.get()
print(f"Inclusive scan custom result: {result}")

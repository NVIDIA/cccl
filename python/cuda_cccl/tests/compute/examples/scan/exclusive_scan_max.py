# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Exclusive scan using custom maximum operation.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Define the binary operation for the scan.


def max_op(a, b):
    return max(a, b)


# Prepare the input and output arrays.
h_init = np.array([1], dtype="int32")
d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
d_output = cp.empty_like(d_input, dtype="int32")

# Perform the exclusive scan.
scanner = cuda.compute.make_exclusive_scan(d_input, d_output, max_op, h_init)
temp_storage_bytes = int(
    scanner.get_temp_storage_bytes(
        d_input,
        d_output,
        d_input.size,
        init_value=h_init,
        op=max_op,
    )
)
d_temp_storage = None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
scanner.compute(
    d_temp_storage,
    d_input,
    d_output,
    d_input.size,
    init_value=h_init,
    op=max_op,
)

# Verify the result.
expected = np.asarray([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
result = d_output.get()

np.testing.assert_equal(result, expected)
print(f"Exclusive scan max result: {result}")

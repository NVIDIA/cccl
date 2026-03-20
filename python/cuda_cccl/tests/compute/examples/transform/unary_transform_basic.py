# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use unary_transform to apply a unary operation to each element.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and output arrays.
input_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
d_in = cp.asarray(input_data)
d_out = cp.empty_like(d_in)


# Define the unary operation.
def op(a):
    return a + 1


# Perform the unary transform.
transformer = cuda.compute.make_unary_transform(d_in, d_out, op)
temp_storage_bytes = int(
    transformer.get_temp_storage_bytes(
        d_in,
        d_out,
        op,
        len(d_in),
    )
)
d_temp_storage = None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
transformer.compute(
    d_temp_storage,
    d_in,
    d_out,
    op,
    len(d_in),
)

# Verify the result.
result = d_out.get()
expected = input_data + 1

np.testing.assert_array_equal(result, expected)
print(f"Unary transform result: {result}")

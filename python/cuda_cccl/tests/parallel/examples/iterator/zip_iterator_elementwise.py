# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use zip_iterator to perform elementwise sum of two arrays.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input arrays.
d_input1 = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
d_input2 = cp.array([10, 20, 30, 40, 50], dtype=np.int32)

# Create the zip iterator.
zip_it = parallel.ZipIterator(d_input1, d_input2)

# Prepare the output array.
num_items = len(d_input1)
d_output = cp.empty(num_items, dtype=np.int32)


def sum_paired_values(pair):
    """Extract values from the zip iterator pair and sum them."""
    return pair[0] + pair[1]


# Perform the unary transform.
parallel.unary_transform(zip_it, d_output, sum_paired_values, num_items)

# Calculate the expected results.
expected = d_input1.get() + d_input2.get()
result = d_output.get()

# Verify the result.
np.testing.assert_allclose(result, expected)

print(f"Input array 1: {d_input1.get()}")
print(f"Input array 2: {d_input2.get()}")
print(f"Elementwise sum result: {result}")
print(f"Expected result: {expected}")

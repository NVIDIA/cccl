# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Demonstrate transform with permutation iterator as output (scatter operation).
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    CountingIterator,
    PermutationIterator,
)


def square_op(x):
    return x * x


# Prepare the output values array and indices.
d_values = cp.zeros(10, dtype=np.int32)  # Output array
d_indices = cp.asarray([9, 3, 7, 1, 5], dtype=np.int32)  # Scatter indices

# Create input iterator that generates: 0, 1, 2, 3, 4
input_it = CountingIterator(np.int32(0))

# Create permutation iterator for output (scatter).
# This will write to: values[9], values[3], values[7], values[1], values[5]
perm_it = PermutationIterator(d_values, d_indices)

# Perform the transform, scattering squared values to permuted locations.
num_items = len(d_indices)
cuda.compute.unary_transform(input_it, perm_it, square_op, num_items)

# Verify the result: values[9]=0, values[3]=1, values[7]=4, values[1]=9, values[5]=16
# Other positions should remain 0
expected = np.zeros(10, dtype=np.int32)
for i, idx in enumerate(d_indices.get()):
    expected[idx] = i * i

assert np.array_equal(d_values.get(), expected)
print(f"Permutation output iterator result: {d_values.get()}")
print(f"Expected: {expected}")

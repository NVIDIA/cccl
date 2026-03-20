# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Using ShuffleIterator to obtain a random permutation of an array
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import PermutationIterator, ShuffleIterator

# Input data and output array:
d_input = cp.asarray([10, 20, 30, 40, 50, 60, 70, 80, 100], dtype=np.int32)
d_output = cp.empty_like(d_input)
num_items = len(d_input)

# Create a shuffle iterator that produces a random permutation of [0, num_items)
shuffle_it = ShuffleIterator(num_items, seed=42)

# Use PermutationIterator to permute the data according to the random indices
perm_it = PermutationIterator(d_input, shuffle_it)

identity_op = lambda x: x
transformer = cuda.compute.make_unary_transform(perm_it, d_output, identity_op)
temp_storage_bytes = int(
    transformer.get_temp_storage_bytes(
        perm_it, d_output, identity_op, num_items
    )
)
d_temp_storage = None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
transformer.compute(
    d_temp_storage,
    perm_it,
    d_output,
    identity_op,
    num_items,
)

# Verify it is a valid permutation of the input data:
cp.testing.assert_array_equal(cp.sort(d_output), d_input)

# Print the values
print(f"Input data: {d_input}")
print(f"Shuffled data: {d_output}")

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Demonstrate reduction with permutation iterator as input.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
    PermutationIterator,
)

# Create a permutation iterator which selects values at the given indices:
d_values = cp.asarray([10, 20, 30, 40, 50], dtype=np.int32)
d_indices = cp.asarray([2, 0, 4, 1], dtype=np.int32)  # permutation indices
perm_it = PermutationIterator(d_values, d_indices)

# Prepare the initial value and output for the reduction.
h_init = np.array([0], dtype=np.int32)
d_output = cp.empty(1, dtype=np.int32)

# Perform the reduction on the permuted values.
num_items = len(d_indices)
cuda.compute.reduce_into(perm_it, d_output, OpKind.PLUS, num_items, h_init)

# Verify the result:
expected_output = d_values[d_indices].sum()
assert d_output[0] == expected_output
print(f"Permutation iterator result: {d_output[0]} (expected: {expected_output})")

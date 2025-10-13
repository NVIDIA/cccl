# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Demonstrate composed permutation iterator with transform iterator.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    PermutationIterator,
    TransformIterator,
)


def square_op(x):
    return x * x


# Create a CountingIterator that generates: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
counting_it = CountingIterator(np.int32(0))

# Wrap it in a TransformIterator to square the values: 0, 1, 4, 9, 16, 25, 36, 49, 64, 81
transform_it = TransformIterator(counting_it, square_op)

# Create indices to permute the squared values
d_indices = cp.asarray([3, 1, 5, 2], dtype=np.int32)

# Create permutation iterator that accesses the squared counting iterator
# This will access: squares[3]=9, squares[1]=1, squares[5]=25, squares[2]=4
perm_it = PermutationIterator(transform_it, d_indices)

# Prepare the initial value and output for the reduction
h_init = np.array([0], dtype=np.int32)
d_output = cp.empty(1, dtype=np.int32)

# Perform the reduction on the composed iterator
num_items = len(d_indices)
cuda.compute.reduce_into(perm_it, d_output, OpKind.PLUS, num_items, h_init)

# Verify the result: 9 + 1 + 25 + 4 = 39
expected_output = 9 + 1 + 25 + 4
assert d_output[0] == expected_output
print(
    f"Composed permutation iterator result: {d_output[0]} (expected: {expected_output})"
)

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Demonstrate reduction with transform iterator.
"""

import functools

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def transform_op(a):
    return -a if a % 2 == 0 else a


# Prepare the input and output arrays.
first_item = 10
num_items = 100

transform_it = parallel.TransformIterator(
    parallel.CountingIterator(np.int32(first_item)), transform_op
)  # Input sequence
h_init = np.array([0], dtype=np.int64)  # Initial value for the reduction
d_output = cp.empty(1, dtype=np.int64)  # Storage for output

# Perform the reduction.
parallel.reduce_into(transform_it, d_output, parallel.OpKind.PLUS, num_items, h_init)

# Verify the result.
expected_output = functools.reduce(
    lambda a, b: a + b,
    [-a if a % 2 == 0 else a for a in range(first_item, first_item + num_items)],
)

# Test assertions
print(f"Transform iterator result: {d_output[0]} (expected: {expected_output})")
assert (d_output == expected_output).all()
assert d_output[0] == expected_output

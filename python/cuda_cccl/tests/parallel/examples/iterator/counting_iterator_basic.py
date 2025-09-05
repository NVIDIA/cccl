# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use counting_iterator.
"""

import functools

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input and output arrays.
first_item = 10
num_items = 3

# Create the counting iterator.
first_it = parallel.CountingIterator(np.int32(first_item))

# Prepare the initial value for the reduction.
h_init = np.array([0], dtype=np.int32)

# Prepare the output array.
d_output = cp.empty(1, dtype=np.int32)

# Perform the reduction.
parallel.reduce_into(first_it, d_output, parallel.OpKind.PLUS, num_items, h_init)

# Verify the result.
expected_output = functools.reduce(
    lambda a, b: a + b, range(first_item, first_item + num_items)
)
assert (d_output == expected_output).all()
print(f"Counting iterator result: {d_output[0]} (expected: {expected_output})")

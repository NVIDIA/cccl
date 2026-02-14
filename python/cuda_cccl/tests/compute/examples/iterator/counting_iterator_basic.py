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

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
)

# Prepare the input and output arrays.
first_item = 1
num_items = 100

# Create the counting iterator.
first_it = CountingIterator(np.int32(first_item))

# Prepare the initial value for the reduction.
h_init = np.array([0], dtype=np.int32)

# Prepare the output array.
d_output = cp.empty(1, dtype=np.int32)

# Perform the reduction.
cuda.compute.reduce_into(first_it, d_output, OpKind.PLUS, num_items, h_init)

# Verify the result.
expected_output = functools.reduce(
    lambda a, b: a + b, range(first_item, first_item + num_items)
)
assert (d_output == expected_output).all()
print(f"Counting iterator result: {d_output[0]} (expected: {expected_output})")

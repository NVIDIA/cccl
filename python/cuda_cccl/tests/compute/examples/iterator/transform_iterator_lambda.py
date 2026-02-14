# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Demonstrate TransformIterator with a lambda function.

This example shows how to use a lambda function with TransformIterator
to apply a transformation on-the-fly during reduction, without needing
to define a named function.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    TransformIterator,
)

# Prepare the parameters.
first_item = 1
num_items = 10

# Create a TransformIterator that squares each value from a CountingIterator
# using a lambda function.
transform_it = TransformIterator(
    CountingIterator(np.int32(first_item)), lambda x: x * x
)

h_init = np.array([0], dtype=np.int32)
d_output = cp.empty(1, dtype=np.int32)

# Perform the reduction: sum of squares from 1 to 10.
cuda.compute.reduce_into(transform_it, d_output, OpKind.PLUS, num_items, h_init)

# Verify the result: 1^2 + 2^2 + ... + 10^2 = 385
expected_output = sum(x * x for x in range(first_item, first_item + num_items))
assert d_output[0] == expected_output
print(
    f"Sum of squares with lambda TransformIterator: {d_output[0]} (expected: {expected_output})"
)

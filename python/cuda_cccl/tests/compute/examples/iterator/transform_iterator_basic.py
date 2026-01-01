# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Using ``reduce_into`` with a ``TransformIterator`` to compute the
sum of squares of a sequence of numbers.
"""

import cupy as cp
import numpy as np

from cuda.compute import (
    OpKind,
    TransformIterator,
    reduce_into,
)

# Prepare the input and output arrays.
d_input = cp.arange(10, dtype=np.int32)
d_output = cp.empty(1, dtype=np.int32)
h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction

# Create a TransformIterator to (lazily) apply the square
it_input = TransformIterator(d_input, lambda a: a**2)

# Use `reduce_into` to compute the sum of the squares of the input.
reduce_into(it_input, d_output, OpKind.PLUS, len(d_input), h_init)

# Verify the result.
expected_output = cp.sum(d_input**2).get()
assert d_output[0] == expected_output
print(f"Transform iterator result: {d_output[0]} (expected: {expected_output})")

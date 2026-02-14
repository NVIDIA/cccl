# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use constant_iterator.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    ConstantIterator,
    OpKind,
)

# Prepare the input and output arrays.
constant_value = 42
num_items = 5

# Create the constant iterator.
constant_it = ConstantIterator(np.int32(constant_value))

# Prepare the initial value for the reduction.
h_init = np.array([0], dtype=np.int32)

# Prepare the output array.
d_output = cp.empty(1, dtype=np.int32)

# Perform the reduction.
cuda.compute.reduce_into(constant_it, d_output, OpKind.PLUS, num_items, h_init)

# Verify the result.
expected_output = constant_value * num_items
assert (d_output == expected_output).all()
print(f"Constant iterator result: {d_output[0]} (expected: {expected_output})")

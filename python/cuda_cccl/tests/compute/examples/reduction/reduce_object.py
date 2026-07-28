# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Reduction example using the object API.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
)

# Prepare the input and output arrays.
dtype = np.int32
init_value = 5
h_init = np.array([init_value], dtype=dtype)
h_input = np.array([1, 2, 3, 4], dtype=dtype)
d_input = cp.asarray(h_input)
d_output = cp.empty(1, dtype=dtype)

# Create a reducer object.
reducer = cuda.compute.make_reduce_into(
    d_in=d_input, d_out=d_output, op=OpKind.PLUS, h_init=h_init
)

# Get the temporary storage size.
temp_storage_size = reducer(
    temp_storage=None,
    d_in=d_input,
    d_out=d_output,
    num_items=len(h_input),
    op=OpKind.PLUS,
    h_init=h_init,
)

# Allocate temporary storage using any user-defined allocator.
# The result must be an object exposing `__cuda_array_interface__`.
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Perform the reduction.
reducer(
    temp_storage=d_temp_storage,
    d_in=d_input,
    d_out=d_output,
    num_items=len(h_input),
    op=OpKind.PLUS,
    h_init=h_init,
)

expected_result = np.sum(h_input) + init_value
actual_result = d_output.get()[0]
assert actual_result == expected_result
print("Reduce object example completed successfully")

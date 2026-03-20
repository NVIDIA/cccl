# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Computing the minimum value of a sequence using `reduce_into`.
"""

import cupy as cp
import numpy as np

import cuda.compute


def min_op(a, b):
    # the binary operation for the reduction
    return a if a < b else b


# Prepare the input and output arrays.
dtype = np.int32
h_init = np.array([42], dtype=dtype)
d_input = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=dtype)
d_output = cp.empty(1, dtype=dtype)

# Perform the reduction.
reducer = cuda.compute.make_reduce_into(d_input, d_output, min_op, h_init)
temp_storage_bytes = int(
    reducer.get_temp_storage_bytes(
        d_input,
        d_output,
        len(d_input),
        h_init=h_init,
        op=min_op,
    )
)
d_temp_storage = None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
reducer.compute(
    d_temp_storage,
    d_input,
    d_output,
    len(d_input),
    h_init=h_init,
    op=min_op,
)

# Verify the result.
expected_output = 0
result = d_output.get()[0]

assert result == expected_output
print(f"Min reduction result: {result}")

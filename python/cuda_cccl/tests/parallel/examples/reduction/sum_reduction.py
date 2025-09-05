# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Sum all values in an array using reduction with PLUS operation.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input and output arrays.
dtype = np.int32
h_init = np.array([0], dtype=dtype)
d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
d_output = cp.empty(1, dtype=dtype)

# Perform the reduction.
parallel.reduce_into(d_input, d_output, parallel.OpKind.PLUS, len(d_input), h_init)

# Verify the result.
expected_output = 15
assert (d_output == expected_output).all()
result = d_output[0]
print(f"Sum reduction result: {result}")

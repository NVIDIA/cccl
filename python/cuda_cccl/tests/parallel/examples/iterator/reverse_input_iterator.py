# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use reverse_input_iterator.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input and output arrays.
h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
d_input = cp.asarray(h_input)

# Create the reverse input iterator.
reverse_it = parallel.ReverseIterator(d_input)
d_output = cp.empty(len(d_input), dtype=np.int32)

# Prepare the initial value for the reduction.
h_init = np.array(0, dtype=np.int32)

# Perform the reduction.
parallel.inclusive_scan(
    reverse_it, d_output, parallel.OpKind.PLUS, h_init, len(d_input)
)

# Verify the result.
expected_output = np.array([5, 9, 12, 14, 15], dtype=np.int32)
result = d_output.get()

np.testing.assert_array_equal(result, expected_output)
print(f"Original input: {h_input}")
print(f"Reverse scan result: {result}")
print(f"Expected result: {expected_output}")

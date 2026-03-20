# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use reverse_output_iterator.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
    ReverseIterator,
)

# Prepare the input and output arrays.
h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
d_input = cp.asarray(h_input)

# Prepare the output array.
d_output = cp.empty(len(d_input), dtype=np.int32)
h_init = np.array(0, dtype=np.int32)

# Create the reverse output iterator.
reverse_out_it = ReverseIterator(d_output)

# Perform the scan using the multi-step object API.
scanner = cuda.compute.make_inclusive_scan(d_input, reverse_out_it, OpKind.PLUS, h_init)
temp_storage_bytes = int(
    scanner.get_temp_storage_bytes(
        d_input,
        reverse_out_it,
        len(d_input),
        init_value=h_init,
        op=OpKind.PLUS,
    )
)
d_temp_storage = None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
scanner.compute(
    d_temp_storage,
    d_input,
    reverse_out_it,
    len(d_input),
    init_value=h_init,
    op=OpKind.PLUS,
)

# Verify the result.
expected_output = np.array([15, 10, 6, 3, 1], dtype=np.int32)
result = d_output.get()

np.testing.assert_array_equal(result, expected_output)
print(f"Original input: {h_input}")
print(f"Reverse output result: {result}")
print(f"Expected result: {expected_output}")

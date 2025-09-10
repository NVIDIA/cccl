# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use segmented_reduce to find the minimum in each segment.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def min_op(a, b):
    return a if a < b else b


dtype = np.dtype(np.int32)
max_val = np.iinfo(dtype).max
h_init = np.asarray(max_val, dtype=dtype)

# Prepare the offsets.
offsets = cp.array([0, 7, 11, 16], dtype=np.int64)
first_segment = (8, 6, 7, 5, 3, 0, 9)
second_segment = (-4, 3, 0, 1)
third_segment = (3, 1, 11, 25, 8)

# Prepare the input array.
d_input = cp.array(
    [*first_segment, *second_segment, *third_segment],
    dtype=dtype,
)

# Prepare the start and end offsets.
start_o = offsets[:-1]
end_o = offsets[1:]

# Prepare the output array.
n_segments = start_o.size
d_output = cp.empty(n_segments, dtype=dtype)

# Perform the segmented reduce.
parallel.segmented_reduce(d_input, d_output, start_o, end_o, min_op, h_init, n_segments)

# Verify the result.
expected_output = cp.asarray([0, -4, 1], dtype=d_output.dtype)
assert (d_output == expected_output).all()
print(f"Segmented reduce basic result: {d_output.get()}")

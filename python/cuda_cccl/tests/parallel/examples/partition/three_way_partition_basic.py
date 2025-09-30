# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use three_way_partition to partition a sequence of integers into three parts.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input and output arrays.
dtype = np.int32
h_input = np.array([0, 2, 9, 1, 5, 6, 7, -3, 17, 10], dtype=dtype)
d_input = cp.asarray(h_input)
d_first_part = cp.empty_like(d_input)
d_second_part = cp.empty_like(d_input)
d_unselected = cp.empty_like(d_input)
d_num_selected = cp.empty(2, dtype=np.int64)


def less_than_op(x):
    return x < 8 and x > 0


def greater_than_equal_op(x):
    return x >= 8


# Perform the three_way_partition.
parallel.three_way_partition(
    d_input,
    d_first_part,
    d_second_part,
    d_unselected,
    d_num_selected,
    less_than_op,
    greater_than_equal_op,
    len(h_input),
)

# Verify the result.
expected_first_part = np.array([0, 2, 1, 5, 6, 7], dtype=dtype)
expected_second_part = np.array([9, 17, 10], dtype=dtype)
expected_unselected = np.array([-3], dtype=dtype)
expected_num_selected = np.array([6, 3], dtype=np.int64)
actual_first_part = d_first_part.get()
actual_second_part = d_second_part.get()
actual_unselected = d_unselected.get()
actual_num_selected = d_num_selected.get()

np.testing.assert_array_equal(actual_first_part, expected_first_part)
np.testing.assert_array_equal(actual_second_part, expected_second_part)
np.testing.assert_array_equal(actual_unselected, expected_unselected)
np.testing.assert_array_equal(actual_num_selected, expected_num_selected)

print("Three way partition basic example completed successfully")

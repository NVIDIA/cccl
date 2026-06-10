# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use three_way_partition with the object API.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the input and output arrays.
dtype = np.int32
h_input = np.array([0, 2, 9, 1, 5, 6, 7, -3, 17, 10], dtype=dtype)
d_input = cp.asarray(h_input)
d_first_part = cp.empty_like(d_input)
d_second_part = cp.empty_like(d_input)
d_unselected = cp.empty_like(d_input)
d_num_selected = cp.empty(2, dtype=np.int64)


def less_than_op(x):
    return x < 8 and x >= 0


def greater_than_equal_op(x):
    return x >= 8


# Create the three_way_partition object.
partitioner = cuda.compute.make_three_way_partition(
    d_in=d_input,
    d_first_part_out=d_first_part,
    d_second_part_out=d_second_part,
    d_unselected_out=d_unselected,
    d_num_selected_out=d_num_selected,
    select_first_part_op=less_than_op,
    select_second_part_op=greater_than_equal_op,
)

# Get the temporary storage size.
temp_storage_size = partitioner(
    temp_storage=None,
    d_in=d_input,
    d_first_part_out=d_first_part,
    d_second_part_out=d_second_part,
    d_unselected_out=d_unselected,
    d_num_selected_out=d_num_selected,
    select_first_part_op=less_than_op,
    select_second_part_op=greater_than_equal_op,
    num_items=len(h_input),
)
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Perform the three_way_partition.
partitioner(
    temp_storage=d_temp_storage,
    d_in=d_input,
    d_first_part_out=d_first_part,
    d_second_part_out=d_second_part,
    d_unselected_out=d_unselected,
    d_num_selected_out=d_num_selected,
    select_first_part_op=less_than_op,
    select_second_part_op=greater_than_equal_op,
    num_items=len(h_input),
)

# Verify the result.
expected_first_part = np.array([0, 2, 1, 5, 6, 7], dtype=dtype)
expected_second_part = np.array([9, 17, 10], dtype=dtype)
expected_unselected = np.array([-3], dtype=dtype)
expected_num_selected = np.array([6, 3], dtype=np.int64)

actual_num_selected = d_num_selected.get()
num_selected_first_part = int(actual_num_selected[0])
num_selected_second_part = int(actual_num_selected[1])
actual_first_part = d_first_part.get()[:num_selected_first_part]
actual_second_part = d_second_part.get()[:num_selected_second_part]
actual_unselected = d_unselected.get()[
    : d_input.size - num_selected_first_part - num_selected_second_part
]

np.testing.assert_array_equal(actual_first_part, expected_first_part)
np.testing.assert_array_equal(actual_second_part, expected_second_part)
np.testing.assert_array_equal(actual_unselected, expected_unselected)
np.testing.assert_array_equal(actual_num_selected, expected_num_selected)

print("Three way partition object example completed successfully")

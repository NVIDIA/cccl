# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
import cupy as cp

from cuda.compute.algorithms import select

# Create input data
d_in = cp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=cp.int32)
d_out = cp.empty_like(d_in)
d_num_selected = cp.zeros(2, dtype=cp.uint64)


# Define select condition (keep even numbers)
def is_even(x):
    return x % 2 == 0


# Execute select
select(d_in, d_out, d_num_selected, is_even, len(d_in))

# Get results
num_selected = int(d_num_selected[0])
result = d_out[:num_selected].get()
print(f"Selected {num_selected} items: {result}")
# Output: Selected 4 items: [2 4 6 8]
# example-end

assert num_selected == 4
assert (result == [2, 4, 6, 8]).all()

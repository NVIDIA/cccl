# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
import cupy as cp

from cuda.compute.algorithms import make_select

# Create input data
d_in = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=cp.int32)
d_out = cp.empty_like(d_in)
d_num_selected = cp.zeros(2, dtype=cp.uint64)


# Define select condition (keep values > 5)
def greater_than_5(x):
    return x > 5


# Create select object (can be reused)
selector = make_select(d_in, d_out, d_num_selected, greater_than_5)

# Get required temp storage
temp_storage_bytes = selector(
    None, d_in, d_out, d_num_selected, greater_than_5, len(d_in)
)
d_temp_storage = cp.empty(temp_storage_bytes, dtype=cp.uint8)

# Execute select
selector(d_temp_storage, d_in, d_out, d_num_selected, greater_than_5, len(d_in))

# Get results
num_selected = int(d_num_selected[0])
result = d_out[:num_selected].get()
print(f"Selected {num_selected} items: {result}")
# Output: Selected 5 items: [ 6  7  8  9 10]

# Reuse the same select object with different input
d_in2 = cp.array([10, 20, 3, 15, 2, 8, 30], dtype=cp.int32)
d_out2 = cp.empty_like(d_in2)
d_num_selected2 = cp.zeros(2, dtype=cp.uint64)

selector(d_temp_storage, d_in2, d_out2, d_num_selected2, greater_than_5, len(d_in2))

num_selected2 = int(d_num_selected2[0])
result2 = d_out2[:num_selected2].get()
print(f"Second select: {num_selected2} items: {result2}")
# Output: Second select: 5 items: [10 20 15  8 30]
# example-end

assert num_selected == 5
assert (result == [6, 7, 8, 9, 10]).all()

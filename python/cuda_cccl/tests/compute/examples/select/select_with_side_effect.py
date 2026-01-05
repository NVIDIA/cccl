# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
import cupy as cp
from numba import cuda as numba_cuda

from cuda.compute.algorithms import select

# Create input data: values 0 to 99
d_in = cp.arange(100, dtype=cp.int32)
d_out = cp.empty_like(d_in)
d_num_selected = cp.empty(1, dtype=cp.uint64)

# Counter for rejected items (side effect state)
reject_count = cp.zeros(1, dtype=cp.int32)


# Define condition that counts rejected items as a side effect
def count_rejects(x):
    if x % 2 == 0:
        return True
    else:
        numba_cuda.atomic.add(reject_count, 0, 1)
        return False


# Execute select - selects even numbers, counts rejections
select(d_in, d_out, d_num_selected, count_rejects, len(d_in))

# Get results
num_selected = int(d_num_selected.get()[0])
num_rejected = int(reject_count.get()[0])
result = d_out[:num_selected].get()

print(f"Selected {num_selected} items (values % 2 == 0)")
print(f"Rejected {num_rejected} items (values % 2 != 0)")
print(f"First 5 selected: {result[:5]}")
# Output:
# Selected 50 items (even numbers)
# Rejected 50 items (odd numbers)
# First 5 selected: [0 2 4 6 8]
# example-end

assert num_selected == 50  # Even numbers
assert num_rejected == 50  # Odd numbers

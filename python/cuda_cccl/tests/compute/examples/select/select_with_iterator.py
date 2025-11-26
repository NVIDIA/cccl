# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
import cupy as cp

from cuda.compute.algorithms import select
from cuda.compute.iterators import TransformIterator

# Create input data
d_in = cp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=cp.int32)
d_out = cp.empty_like(d_in)
d_num_selected = cp.zeros(2, dtype=cp.uint64)


# Create iterator that squares each value
def square(x):
    return x * x


squared_iter = TransformIterator(d_in, square)


# Select squared values that are greater than 20
def greater_than_20(x):
    return x > 20


select(squared_iter, d_out, d_num_selected, greater_than_20, len(d_in))

# Get results
num_selected = int(d_num_selected[0])
result = d_out[:num_selected].get()
print(f"Selected {num_selected} items: {result}")
# Output: Selected 4 items: [25 36 49 64]
# (5^2=25, 6^2=36, 7^2=49, 8^2=64, all > 20)
# example-end

assert num_selected == 4
assert (result == [25, 36, 49, 64]).all()

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Simultaneously computing the minimum and maximum values of a sequence using `reduce_into`
with a custom data type defined as a numpy structured dtype.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import TransformIterator

# Define a custom data type for the accumulator using numpy structured dtype.
minmax_dtype = np.dtype([("min_val", np.float64), ("max_val", np.float64)], align=True)


# Define the binary operation for the reduction.
# Type annotations use the numpy dtype; return tuple is implicitly converted
def minmax_op(v1: minmax_dtype, v2: minmax_dtype) -> minmax_dtype:
    c_min = min(v1.min_val, v2.min_val)
    c_max = max(v1.max_val, v2.max_val)
    return (c_min, c_max)


# Define a transform operation to convert a value `x` to (abs(x), abs(x)).
def transform_op(v) -> minmax_dtype:
    av = abs(v)
    return (av, av)


# Prepare the input and output data.
nelems = 4096
d_in = cp.random.randn(nelems)
tr_it = TransformIterator(d_in, transform_op)

d_out = cp.empty(tuple(), dtype=minmax_dtype)

h_init = np.void((np.inf, -np.inf), dtype=minmax_dtype)

# Perform the reduction.
cuda.compute.reduce_into(tr_it, d_out, minmax_op, nelems, h_init)

# Verify the result.
actual = d_out.get()
h = np.abs(d_in.get())
expected = np.asarray([(h.min(), h.max())], dtype=minmax_dtype)

assert actual == expected
print(f"MinMax reduction result: {actual}")

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Simultaneously computing the minimum and maximum values of a sequence using `reduce_into`
with a custom data type.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    TransformIterator,
    gpu_struct,
)


# Define a custom data type for the accumulator.
@gpu_struct
class MinMax:
    min_val: np.float64
    max_val: np.float64


# Define the binary operation for the reduction.
def minmax_op(v1: MinMax, v2: MinMax):
    c_min = min(v1.min_val, v2.min_val)
    c_max = max(v1.max_val, v2.max_val)
    return MinMax(c_min, c_max)


# Define a transform operation to convert a value `x` to MinMax(abs(x), abs(x)).
def transform_op(v):
    av = abs(v)
    return MinMax(av, av)


# Prepare the input and output data.
nelems = 4096
d_in = cp.random.randn(nelems)
tr_it = TransformIterator(d_in, transform_op)

d_out = cp.empty(tuple(), dtype=MinMax.dtype)

h_init = MinMax(np.inf, -np.inf)

# Perform the reduction.
cuda.compute.reduce_into(tr_it, d_out, minmax_op, nelems, h_init)

# Verify the result.
actual = d_out.get()
h = np.abs(d_in.get())
expected = np.asarray([(h.min(), h.max())], dtype=MinMax.dtype)

assert actual == expected
print(f"MinMax reduction result: {actual}")

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
TransformOutputIterator example demonstrating reduction with transform output iterator.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Create input and output arrays
d_input = cp.array([1, 2, 3, 4, 5.0], dtype=np.float32)
d_output = cp.empty(shape=1, dtype=np.float32)


# Define the transform operation to be applied
# to the result of the sum reduction
def sqrt(x):
    return x**0.5


# Create transform output iterator
d_out_it = parallel.TransformIterator(d_output, sqrt)


# Apply a sum reduction into the transform output iterator
parallel.reduce_into(
    d_input,
    d_out_it,
    parallel.OpKind.PLUS,
    len(d_input),
    np.asarray([0], dtype=np.float32),
)

assert cp.allclose(d_output, cp.sqrt(cp.sum(d_input)), atol=1e-6)

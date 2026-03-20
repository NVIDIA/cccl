# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
TransformOutputIterator example demonstrating reduction with transform output iterator.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
    TransformOutputIterator,
)

# Create input and output arrays
d_input = cp.array([1, 2, 3, 4, 5.0], dtype=np.float32)
d_output = cp.empty(shape=1, dtype=np.float32)


# Define the transform operation to be applied
# to the result of the sum reduction.
# TransformOutputIterator requires type annotations:
def sqrt(x: np.float32) -> np.float32:
    return x**0.5


# Create transform output iterator
d_out_it = TransformOutputIterator(d_output, sqrt)


# Apply a sum reduction into the transform output iterator
reducer = cuda.compute.make_reduce_into(
    d_input,
    d_out_it,
    OpKind.PLUS,
    np.asarray([0], dtype=np.float32),
)
temp_storage_bytes = int(
    reducer.get_temp_storage_bytes(
        d_input,
        d_out_it,
        len(d_input),
        h_init=np.asarray([0], dtype=np.float32),
        op=OpKind.PLUS,
    )
)
d_temp_storage = (
    None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
)
reducer.compute(
    d_temp_storage,
    d_input,
    d_out_it,
    len(d_input),
    h_init=np.asarray([0], dtype=np.float32),
    op=OpKind.PLUS,
)

assert cp.allclose(d_output, cp.sqrt(cp.sum(d_input)), atol=1e-6)

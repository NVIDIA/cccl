# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Using ``reduce_into`` with a ``TransformIterator`` to compute the
sum of squares of a sequence of numbers.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
    TransformIterator,
)

# Prepare the input and output arrays.
d_input = cp.arange(10, dtype=np.int32)
d_output = cp.empty(1, dtype=np.int32)
h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction

# Create a TransformIterator to (lazily) apply the square
it_input = TransformIterator(d_input, lambda a: a**2)

reducer = cuda.compute.make_reduce_into(it_input, d_output, OpKind.PLUS, h_init)
temp_storage_bytes = int(
    reducer.get_temp_storage_bytes(
        it_input,
        d_output,
        len(d_input),
        h_init=h_init,
        op=OpKind.PLUS,
    )
)
d_temp_storage = None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
reducer.compute(
    d_temp_storage,
    it_input,
    d_output,
    len(d_input),
    h_init=h_init,
    op=OpKind.PLUS,
)

# Verify the result.
expected_output = cp.sum(d_input**2).get()
assert d_output[0] == expected_output
print(f"Transform iterator result: {d_output[0]} (expected: {expected_output})")

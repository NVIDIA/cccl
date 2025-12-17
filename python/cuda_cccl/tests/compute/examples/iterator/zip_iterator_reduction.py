# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use zip_iterator to simultaneously perform a reduction
operation on two arrays.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    ZipIterator,
    gpu_struct,
)


@gpu_struct
class Pair:
    first: np.int32
    second: np.float32


def sum_pairs(p1, p2):
    """Reduction operation that adds corresponding elements of pairs."""
    return Pair(p1[0] + p2[0], p1[1] + p2[1])


# Prepare the input arrays.
d_input1 = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
d_input2 = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

# Create the zip iterator.
zip_it = ZipIterator(d_input1, d_input2)

# Prepare the initial value for the reduction.
h_init = Pair(0, 0.0)

# Prepare the output array.
d_output = cp.empty(1, dtype=Pair.dtype)

# Perform the reduction.
cuda.compute.reduce_into(zip_it, d_output, sum_pairs, len(d_input1), h_init)

# Calculate the expected results.
expected_first = sum(d_input1.get())
expected_second = sum(d_input2.get())

result = d_output.get()[0]
assert result["first"] == expected_first
assert result["second"] == expected_second

print(
    f"Zip iterator result: first={result['first']} (expected: {expected_first}), "
    f"second={result['second']} (expected: {expected_second})"
)

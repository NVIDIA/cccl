# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Run independent direct API calls from multiple Python threads.
"""

from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import OpKind


def reduce_values(h_input):
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.asarray(h_input, dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=len(h_input),
        op=OpKind.PLUS,
        h_init=h_init,
    )

    return int(d_output.get()[0])


inputs = [
    np.array([1, 2, 3, 4], dtype=np.int32),
    np.array([5, 6, 7, 8], dtype=np.int32),
]

with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
    results = list(executor.map(reduce_values, inputs))

expected = [int(np.sum(h_input)) for h_input in inputs]
assert results == expected
print(f"Free-threaded direct API results: {results}")

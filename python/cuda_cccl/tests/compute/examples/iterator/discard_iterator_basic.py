# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use DiscardIterator.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    DiscardIterator,
    OpKind,
)

# Prepare the input and output arrays.
h_in_keys = np.array([1, 1, 2, 3, 3, 7, 8, 8], dtype="int32")
d_in_keys = cp.asarray(h_in_keys)
d_out_keys = cp.empty_like(d_in_keys)
d_out_num_selected = cp.empty(1, np.int32)

# Prepare the discard iterator for values.
d_in_values = DiscardIterator()
d_out_values = DiscardIterator()

# Perform the unique by key operation.
cuda.compute.unique_by_key(
    d_in_keys,
    d_in_values,
    d_out_keys,
    d_out_values,
    d_out_num_selected,
    OpKind.EQUAL_TO,
    d_in_keys.size,
)

# Verify the result.
num_selected = cp.asnumpy(d_out_num_selected)[0]
h_out_keys = cp.asnumpy(d_out_keys)[:num_selected]

expected_keys = np.array([1, 2, 3, 7, 8])

assert np.array_equal(h_out_keys, expected_keys)
print(f"Discard iterator result - keys: {h_out_keys}, count: {num_selected}")

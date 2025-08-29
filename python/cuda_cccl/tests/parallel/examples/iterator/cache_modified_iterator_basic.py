# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use cache_modified_iterator.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input array.
h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
d_input = cp.asarray(h_input)

# Create the cache modified iterator.
cache_it = parallel.CacheModifiedInputIterator(d_input, "stream")

# Prepare the initial value for the reduction.
h_init = np.array([0], dtype=np.int32)

# Prepare the output array.
d_output = cp.empty(1, dtype=np.int32)

# Perform the reduction.
parallel.reduce_into(cache_it, d_output, parallel.OpKind.PLUS, len(d_input), h_init)

# Verify the result.
expected_output = sum(h_input)  # 1 + 2 + 3 + 4 + 5 = 15
assert (d_output == expected_output).all()
print(f"Cache modified iterator result: {d_output[0]} (expected: {expected_output})")

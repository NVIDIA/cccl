# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use zip_iterator with counting iterator to
find the index with maximum value in an array.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


@parallel.gpu_struct
class IndexValuePair:
    index: np.int32
    value: np.int32


def max_by_value(p1, p2):
    """Reduction operation that returns the pair with the larger value."""
    return p1 if p1[1] > p2[1] else p2


# Create the counting iterator.
counting_it = parallel.CountingIterator(np.int32(0))

# Prepare the input array.
arr = cp.asarray([0, 1, 2, 4, 7, 3, 5, 6], dtype=np.int32)

# Create the zip iterator.
zip_it = parallel.ZipIterator(counting_it, arr)

num_items = 8
h_init = IndexValuePair(-1, -1)
d_output = cp.empty(1, dtype=IndexValuePair.dtype)

# Perform the reduction.
parallel.reduce_into(zip_it, d_output, max_by_value, num_items, h_init)

result = d_output.get()[0]
expected_index = 4
expected_value = 7

assert result["index"] == expected_index
assert result["value"] == expected_value

print(
    f"Zip iterator with counting result: index={result['index']} "
    f"(expected: {expected_index}), value={result['value']} (expected: {expected_value})"
)

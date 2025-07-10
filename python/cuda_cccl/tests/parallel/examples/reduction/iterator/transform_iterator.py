# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Transform iterator example demonstrating reduction with transform iterator.
"""

import functools

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.cccl.parallel.experimental.iterators as iterators

def transform_iterator_example():
    """Demonstrate reduction with transform iterator."""

    def add_op(a, b):
        return a + b

    def transform_op(a):
        return -a if a % 2 == 0 else a

    first_item = 10
    num_items = 100

    transform_it = iterators.TransformIterator(
        iterators.CountingIterator(np.int32(first_item)), transform_op
    )  # Input sequence
    h_init = np.array([0], dtype=np.int64)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int64)  # Storage for output

    # Instantiate reduction, determine storage requirements, and allocate storage
    reduce_into = algorithms.reduce_into(transform_it, d_output, add_op, h_init)
    temp_storage_size = reduce_into(None, transform_it, d_output, num_items, h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, transform_it, d_output, num_items, h_init)

    expected_output = functools.reduce(
        lambda a, b: a + b,
        [-a if a % 2 == 0 else a for a in range(first_item, first_item + num_items)],
    )

    # Test assertions
    print(f"Transform iterator result: {d_output[0]} (expected: {expected_output})")
    assert (d_output == expected_output).all()
    assert d_output[0] == expected_output
    return d_output[0]


if __name__ == "__main__":
    print("Running transform iterator example...")
    transform_iterator_example()
    print("Transform iterator example completed successfully!")

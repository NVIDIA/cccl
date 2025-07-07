# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Counting iterator example demonstrating reduction with counting iterator.
"""

import functools

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.cccl.parallel.experimental.iterators as iterators


def counting_iterator_example():
    """Demonstrate reduction with counting iterator."""

    def add_op(a, b):
        return a + b

    first_item = 10
    num_items = 3

    first_it = iterators.CountingIterator(np.int32(first_item))  # Input sequence
    h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int32)  # Storage for output

    # Instantiate reduction, determine storage requirements, and allocate storage
    temp_storage_size = algorithms.reduce_into(
        None, first_it, d_output, num_items, add_op, h_init
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    algorithms.reduce_into(
        d_temp_storage, first_it, d_output, num_items, add_op, h_init
    )

    expected_output = functools.reduce(
        lambda a, b: a + b, range(first_item, first_item + num_items)
    )
    assert (d_output == expected_output).all()
    print(f"Counting iterator result: {d_output[0]} (expected: {expected_output})")
    return d_output[0]


if __name__ == "__main__":
    print("Running counting iterator example...")
    counting_iterator_example()
    print("Counting iterator example completed successfully!")

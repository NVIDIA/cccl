# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Counting iterator example demonstrating reduction with counting iterator.
"""

import functools

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def counting_iterator_example():
    """Demonstrate reduction with counting iterator."""

    first_item = 10
    num_items = 3

    first_it = parallel.CountingIterator(np.int32(first_item))  # Input sequence
    h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int32)  # Storage for output

    # Run reduction
    parallel.reduce_into(first_it, d_output, parallel.OpKind.PLUS, num_items, h_init)

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

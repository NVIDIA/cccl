# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Basic reduction examples.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def sum_reduction_example():
    """Sum all values in an array using reduction with PLUS operation."""

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction
    parallel.reduce_into(d_input, d_output, parallel.OpKind.PLUS, len(d_input), h_init)

    expected_output = 15  # 1+2+3+4+5
    assert (d_output == expected_output).all()
    print(f"Sum: {d_output[0]}")
    return d_output[0]


def sum_custom_reduction():
    """Sum only even values in an array using reduction with custom operation."""

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    def add_op(a, b):
        return (a if a % 2 == 0 else 0) + (b if b % 2 == 0 else 0)

    # Run reduction with well-known PLUS operation
    parallel.reduce_into(d_input, d_output, add_op, len(d_input), h_init)

    expected_output = 6  # 2+4
    assert (d_output == expected_output).all()
    print(f"Sum of even values: {d_output[0]}")
    return d_output[0]


if __name__ == "__main__":
    print("Running basic reduction examples...")
    sum_reduction_example()
    sum_custom_reduction()
    print("All reduction examples completed successfully!")

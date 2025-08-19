# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Basic reduction examples demonstrating fundamental reduction operations with both
custom operations and well-known operations.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def sum_reduction_example():
    """Sum all values in an array using reduction with custom operation."""

    def add_op(a, b):
        return a + b

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction
    parallel.reduce_into(d_input, d_output, add_op, len(d_input), h_init)

    expected_output = 15  # 1+2+3+4+5
    assert (d_output == expected_output).all()
    print(f"Sum: {d_output[0]}")
    return d_output[0]


def plus_reduction_example():
    """Demonstrate reduction using well-known PLUS operation."""

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known PLUS operation
    parallel.reduce_into(d_input, d_output, parallel.OpKind.PLUS, len(d_input), h_init)

    expected_output = 15  # 1+2+3+4+5
    assert (d_output == expected_output).all()
    print(f"PLUS reduction result: {d_output[0]}")
    return d_output[0]


def minimum_reduction_example():
    """Demonstrate reduction using well-known MINIMUM operation."""

    dtype = np.int32
    h_init = np.array([1000], dtype=dtype)
    d_input = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known MINIMUM operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.MINIMUM, len(d_input), h_init
    )

    expected_output = 0
    assert (d_output == expected_output).all()
    print(f"MINIMUM reduction result: {d_output[0]}")
    return d_output[0]


def maximum_reduction_example():
    """Demonstrate reduction using well-known MAXIMUM operation."""

    dtype = np.int32
    h_init = np.array([-1000], dtype=dtype)
    d_input = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known MAXIMUM operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.MAXIMUM, len(d_input), h_init
    )

    expected_output = 9
    assert (d_output == expected_output).all()
    print(f"MAXIMUM reduction result: {d_output[0]}")
    return d_output[0]


def multiplies_reduction_example():
    """Demonstrate reduction using well-known MULTIPLIES operation."""

    dtype = np.int32
    h_init = np.array([1], dtype=dtype)
    d_input = cp.array([2, 3, 4], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known MULTIPLIES operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.MULTIPLIES, len(d_input), h_init
    )

    expected_output = 24  # 1*2*3*4
    assert (d_output == expected_output).all()
    print(f"MULTIPLIES reduction result: {d_output[0]}")
    return d_output[0]


def bit_and_reduction_example():
    """Demonstrate reduction using well-known BIT_AND operation."""

    dtype = np.uint32
    h_init = np.array([0xFFFFFFFF], dtype=dtype)  # All bits set
    d_input = cp.array([0xF0F0, 0x00FF, 0xF000], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known BIT_AND operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.BIT_AND, len(d_input), h_init
    )

    expected_output = 0x0000  # 0xF0F0 & 0x00FF & 0xF000
    assert (d_output == expected_output).all()
    print(f"BIT_AND reduction result: 0x{d_output[0]:X}")
    return d_output[0]


def bit_or_reduction_example():
    """Demonstrate reduction using well-known BIT_OR operation."""

    dtype = np.uint32
    h_init = np.array([0x0000], dtype=dtype)
    d_input = cp.array([0x000F, 0x00F0, 0x0F00], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known BIT_OR operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.BIT_OR, len(d_input), h_init
    )

    expected_output = 0x0FFF  # 0x000F | 0x00F0 | 0x0F00
    assert (d_output == expected_output).all()
    print(f"BIT_OR reduction result: 0x{d_output[0]:X}")
    return d_output[0]


if __name__ == "__main__":
    print("Running basic reduction examples...")
    sum_reduction_example()
    plus_reduction_example()
    minimum_reduction_example()
    maximum_reduction_example()
    multiplies_reduction_example()
    bit_and_reduction_example()
    bit_or_reduction_example()
    print("All reduction examples completed successfully!")

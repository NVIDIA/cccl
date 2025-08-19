# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Binary transform examples demonstrating the object API and well-known operations.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def binary_transform_object_example():
    def add_op(a, b):
        return a + b

    dtype = np.int32
    h_input1 = np.array([1, 2, 3, 4], dtype=dtype)
    h_input2 = np.array([10, 20, 30, 40], dtype=dtype)
    d_input1 = cp.asarray(h_input1)
    d_input2 = cp.asarray(h_input2)
    d_output = cp.empty_like(d_input1)

    transformer = parallel.make_binary_transform(d_input1, d_input2, d_output, add_op)

    transformer(d_input1, d_input2, d_output, len(h_input1))

    expected_result = np.array([11, 22, 33, 44], dtype=dtype)
    actual_result = d_output.get()
    print(f"Binary transform object API result: {actual_result}")
    np.testing.assert_array_equal(actual_result, expected_result)
    print("Binary transform object API example passed.")


def plus_binary_transform_example():
    """Demonstrate binary transform using well-known PLUS operation."""

    dtype = np.int32
    d_input1 = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_input2 = cp.array([10, 20, 30, 40, 50], dtype=dtype)
    d_output = cp.empty_like(d_input1, dtype=dtype)

    # Run binary transform with well-known PLUS operation
    parallel.binary_transform(
        d_input1, d_input2, d_output, parallel.OpKind.PLUS, len(d_input1)
    )

    expected_output = np.array([11, 22, 33, 44, 55])
    assert cp.all(d_output.get() == expected_output)
    print(f"PLUS binary transform result: {d_output.get()}")
    return d_output.get()


def multiplies_binary_transform_example():
    """Demonstrate binary transform using well-known MULTIPLIES operation."""

    dtype = np.int32
    d_input1 = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_input2 = cp.array([2, 3, 4, 5, 6], dtype=dtype)
    d_output = cp.empty_like(d_input1, dtype=dtype)

    # Run binary transform with well-known MULTIPLIES operation
    parallel.binary_transform(
        d_input1, d_input2, d_output, parallel.OpKind.MULTIPLIES, len(d_input1)
    )

    expected_output = np.array([2, 6, 12, 20, 30])
    assert cp.all(d_output.get() == expected_output)
    print(f"MULTIPLIES binary transform result: {d_output.get()}")
    return d_output.get()


def maximum_binary_transform_example():
    """Demonstrate binary transform using well-known MAXIMUM operation."""

    dtype = np.int32
    d_input1 = cp.array([1, 5, 2, 8, 3], dtype=dtype)
    d_input2 = cp.array([4, 2, 7, 1, 6], dtype=dtype)
    d_output = cp.empty_like(d_input1, dtype=dtype)

    # Run binary transform with well-known MAXIMUM operation
    parallel.binary_transform(
        d_input1, d_input2, d_output, parallel.OpKind.MAXIMUM, len(d_input1)
    )

    expected_output = np.array([4, 5, 7, 8, 6])
    assert cp.all(d_output.get() == expected_output)
    print(f"MAXIMUM binary transform result: {d_output.get()}")
    return d_output.get()


if __name__ == "__main__":
    print("Running binary transform examples...")
    binary_transform_object_example()
    plus_binary_transform_example()
    multiplies_binary_transform_example()
    maximum_binary_transform_example()
    print("All binary transform examples completed successfully!")

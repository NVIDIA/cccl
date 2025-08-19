# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unary transform examples demonstrating the object API and well-known operations.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def unary_transform_object_example():
    def add_one_op(a):
        return a + 1

    dtype = np.int32
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty_like(d_input)

    transformer = parallel.make_unary_transform(d_input, d_output, add_one_op)

    transformer(d_input, d_output, len(h_input))

    expected_result = np.array([2, 3, 4, 5], dtype=dtype)
    actual_result = d_output.get()
    print(f"Unary transform object API result: {actual_result}")
    np.testing.assert_array_equal(actual_result, expected_result)
    print("Unary transform object API example passed.")


def negate_transform_example():
    """Demonstrate unary transform using well-known NEGATE operation."""

    dtype = np.int32
    d_input = cp.array([1, -2, 3, -4, 5], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    # Run unary transform with well-known NEGATE operation
    parallel.unary_transform(d_input, d_output, parallel.OpKind.NEGATE, len(d_input))

    expected_output = np.array([-1, 2, -3, 4, -5])
    assert cp.all(d_output.get() == expected_output)
    print(f"NEGATE transform result: {d_output.get()}")
    return d_output.get()


def identity_transform_example():
    """Demonstrate unary transform using well-known IDENTITY operation."""

    dtype = np.int32
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    # Run unary transform with well-known IDENTITY operation
    parallel.unary_transform(d_input, d_output, parallel.OpKind.IDENTITY, len(d_input))

    expected_output = np.array([1, 2, 3, 4, 5])
    assert cp.all(d_output.get() == expected_output)
    print(f"IDENTITY transform result: {d_output.get()}")
    return d_output.get()


if __name__ == "__main__":
    print("Running unary transform examples...")
    unary_transform_object_example()
    negate_transform_example()
    identity_transform_example()
    print("All unary transform examples completed successfully!")

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Basic scan examples demonstrating inclusive and exclusive scan operations with both
custom operations and well-known operations.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def inclusive_scan_example():
    """Demonstrate inclusive scan (prefix sum)."""

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    # Run inclusive scan with automatic temp storage allocation
    parallel.inclusive_scan(d_input, d_output, add_op, h_init, d_input.size)

    # Check the result is correct
    expected = np.asarray([-5, -5, -3, -6, -4, 0, 0, -1, 1, 9])
    assert np.array_equal(d_output.get(), expected)
    print(f"Inclusive scan result: {d_output.get()}")
    return d_output.get()


def exclusive_scan_example():
    """Demonstrate exclusive scan with max operation."""

    def max_op(a, b):
        return max(a, b)

    h_init = np.array([1], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    # Run exclusive scan with automatic temp storage allocation
    parallel.exclusive_scan(d_input, d_output, max_op, h_init, d_input.size)

    # Check the result is correct
    expected = np.asarray([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    assert np.array_equal(d_output.get(), expected)
    print(f"Exclusive scan result: {d_output.get()}")
    return d_output.get()


def prefix_sum_example():
    """Simple prefix sum example using inclusive scan."""

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([1, 2, 3, 4, 5], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    # Run inclusive scan with automatic temp storage allocation
    parallel.inclusive_scan(d_input, d_output, add_op, h_init, d_input.size)

    # Check the result is correct (1, 3, 6, 10, 15)
    expected = np.asarray([1, 3, 6, 10, 15])
    assert np.array_equal(d_output.get(), expected)
    print(f"Prefix sum result: {d_output.get()}")
    return d_output.get()


def plus_scan_example():
    """Demonstrate scan using well-known PLUS operation."""

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    # Run exclusive scan with well-known PLUS operation
    parallel.exclusive_scan(
        d_input, d_output, parallel.OpKind.PLUS, h_init, d_input.size
    )

    expected_output = np.array([0, 1, 3, 6, 10])  # exclusive scan
    assert cp.all(d_output.get() == expected_output)
    print(f"PLUS exclusive scan result: {d_output.get()}")

    # Run inclusive scan with well-known PLUS operation
    parallel.inclusive_scan(
        d_input, d_output, parallel.OpKind.PLUS, h_init, d_input.size
    )

    expected_output = np.array([1, 3, 6, 10, 15])  # inclusive scan
    assert cp.all(d_output.get() == expected_output)
    print(f"PLUS inclusive scan result: {d_output.get()}")

    return d_output.get()


def maximum_scan_example():
    """Demonstrate scan using well-known MAXIMUM operation."""

    dtype = np.int32
    h_init = np.array([1], dtype=dtype)
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    # Run exclusive scan with well-known MAXIMUM operation
    parallel.exclusive_scan(
        d_input, d_output, parallel.OpKind.MAXIMUM, h_init, d_input.size
    )

    expected_output = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    assert cp.all(d_output.get() == expected_output)
    print(f"MAXIMUM exclusive scan result: {d_output.get()}")
    return d_output.get()


def multiplies_scan_example():
    """Demonstrate scan using well-known MULTIPLIES operation."""

    dtype = np.int32
    h_init = np.array([1], dtype=dtype)
    d_input = cp.array([2, 3, 4], dtype=dtype)
    d_output = cp.empty_like(d_input, dtype=dtype)

    # Run exclusive scan with well-known MULTIPLIES operation
    parallel.exclusive_scan(
        d_input, d_output, parallel.OpKind.MULTIPLIES, h_init, d_input.size
    )

    expected_output = np.array([1, 2, 6])  # exclusive scan with multiplication
    assert cp.all(d_output.get() == expected_output)
    print(f"MULTIPLIES exclusive scan result: {d_output.get()}")
    return d_output.get()


if __name__ == "__main__":
    print("Running basic scan examples...")
    inclusive_scan_example()
    exclusive_scan_example()
    prefix_sum_example()
    plus_scan_example()
    maximum_scan_example()
    multiplies_scan_example()
    print("All scan examples completed successfully!")

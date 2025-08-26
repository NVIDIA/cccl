# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Basic scan examples demonstrating inclusive and exclusive scan operations.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def inclusive_scan_example():
    """Inclusive scan (prefix sum)."""

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    # Run inclusive scan with automatic temp storage allocation
    parallel.inclusive_scan(
        d_input, d_output, parallel.OpKind.PLUS, h_init, d_input.size
    )

    # Check the result is correct
    expected = np.asarray([-5, -5, -3, -6, -4, 0, 0, -1, 1, 9])
    assert np.array_equal(d_output.get(), expected)
    print(f"Inclusive scan result: {d_output.get()}")
    return d_output.get()


def inclusive_scan_custom_op():
    """Inclusive scan with custom operation (prefix sum of even values)."""

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([1, 2, 3, 4, 5], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    def add_op(a, b):
        return (a if a % 2 == 0 else 0) + (b if b % 2 == 0 else 0)

    parallel.inclusive_scan(d_input, d_output, add_op, h_init, d_input.size)

    expected = np.asarray([0, 2, 2, 6, 6])
    assert np.array_equal(d_output.get(), expected)
    print(f"Inclusive scan result: {d_output.get()}")
    return d_output.get()


def exclusive_scan_example():
    """Exclusive scan (prefix sum)."""

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    parallel.exclusive_scan(
        d_input, d_output, parallel.OpKind.PLUS, h_init, d_input.size
    )

    expected = np.asarray([0, -5, -5, -3, -6, -4, 0, 0, -1, 1])
    assert np.array_equal(d_output.get(), expected)
    print(f"Exclusive scan result: {d_output.get()}")
    return d_output.get()


if __name__ == "__main__":
    print("Running basic scan examples...")
    inclusive_scan_example()
    inclusive_scan_custom_op()
    exclusive_scan_example()
    print("All scan examples completed successfully!")

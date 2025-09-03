# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Merge sort examples demonstrating stable sorting with custom comparison functions.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def basic_merge_sort_example():
    """Demonstrate basic merge sort with keys and values."""

    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)

    # Run merge_sort with automatic temp storage allocation
    parallel.merge_sort(
        d_in_keys,
        d_in_values,
        d_in_keys,
        d_in_values,
        parallel.OpKind.LESS,
        d_in_keys.size,
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_in_keys)
    h_out_values = cp.asnumpy(d_in_values)

    argsort = np.argsort(h_in_keys, stable=True)
    expected_keys = np.array(h_in_keys)[argsort]
    expected_values = np.array(h_in_values)[argsort]

    assert np.array_equal(h_out_keys, expected_keys)
    assert np.array_equal(h_out_values, expected_values)
    print(f"Merge sorted keys: {h_out_keys}")
    print(f"Merge sorted values: {h_out_values}")
    return h_out_keys, h_out_values


def descending_merge_sort_example():
    """Demonstrate merge sort in descending order."""

    def descending_compare_op(lhs, rhs):
        return np.uint8(lhs > rhs)  # Greater than for descending

    h_in_keys = np.array([1, 5, 3, 9, 2, 8, 4, 7, 6], dtype="int32")
    d_in_keys = cp.asarray(h_in_keys)

    # Run merge_sort with automatic temp storage allocation
    parallel.merge_sort(
        d_in_keys, None, d_in_keys, None, descending_compare_op, d_in_keys.size
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_in_keys)
    expected_keys = np.sort(h_in_keys)[::-1]  # Reverse for descending

    assert np.array_equal(h_out_keys, expected_keys)
    print(f"Descending merge sort result: {h_out_keys}")
    return h_out_keys


def string_length_sort_example():
    """Demonstrate merge sort with custom comparison for string lengths."""

    def length_compare_op(lhs, rhs):
        # This would work with string lengths if we had string support
        # For now, demonstrate with array representing string lengths
        return np.uint8(lhs < rhs)

    # Simulate string lengths: ["cat", "elephant", "dog", "a", "bird"]
    h_string_lengths = np.array([3, 8, 3, 1, 4], dtype="int32")
    h_indices = np.array([0, 1, 2, 3, 4], dtype="int32")  # Original indices

    d_lengths = cp.asarray(h_string_lengths)
    d_indices = cp.asarray(h_indices)

    # Sort by string length, keeping track of original indices
    # Run merge_sort with automatic temp storage allocation
    parallel.merge_sort(
        d_lengths, d_indices, d_lengths, d_indices, length_compare_op, d_lengths.size
    )

    # Check the result
    h_out_lengths = cp.asnumpy(d_lengths)
    h_out_indices = cp.asnumpy(d_indices)

    print(
        f"Sorted by length: lengths={h_out_lengths}, original_indices={h_out_indices}"
    )
    return h_out_lengths, h_out_indices


if __name__ == "__main__":
    print("Running merge sort examples...")
    basic_merge_sort_example()
    descending_merge_sort_example()
    string_length_sort_example()
    print("All merge sort examples completed successfully!")

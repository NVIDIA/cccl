# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unique by key examples demonstrating the unique_by_key algorithm.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def basic_unique_by_key_example():
    """Demonstrate basic unique by key operation."""

    h_in_keys = np.array([0, 2, 2, 9, 5, 5, 5, 8], dtype="int32")
    h_in_values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="float32")

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)
    d_out_num_selected = cp.empty(1, np.int32)

    # Run unique_by_key with automatic temp storage allocation
    parallel.unique_by_key(
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        d_out_num_selected,
        parallel.OpKind.EQUAL_TO,
        d_in_keys.size,
    )

    # Check the result is correct
    num_selected = cp.asnumpy(d_out_num_selected)[0]
    h_out_keys = cp.asnumpy(d_out_keys)[:num_selected]
    h_out_values = cp.asnumpy(d_out_values)[:num_selected]

    # Expected: remove consecutive duplicates
    # [0, 2, 2, 9, 5, 5, 5, 8] -> [0, 2, 9, 5, 8]
    # [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 2, 4, 5, 8]
    expected_keys = np.array([0, 2, 9, 5, 8])
    expected_values = np.array([1, 2, 4, 5, 8])

    assert np.array_equal(h_out_keys, expected_keys)
    assert np.array_equal(h_out_values, expected_values)
    print(f"Original keys: {h_in_keys}")
    print(f"Unique keys: {h_out_keys}")
    print(f"Original values: {h_in_values}")
    print(f"Unique values: {h_out_values}")
    print(f"Number of unique elements: {num_selected}")
    return h_out_keys, h_out_values, num_selected


def string_deduplication_example():
    """Demonstrate unique by key for string-like data (using integers to represent strings)."""

    # Simulate string IDs: ["apple", "apple", "banana", "cherry", "cherry", "date"]
    # Using integers to represent string IDs
    h_string_ids = np.array([1, 1, 2, 3, 3, 4], dtype="int32")  # string IDs
    h_frequencies = np.array([5, 3, 8, 2, 7, 1], dtype="int32")  # word frequencies

    d_in_keys = cp.asarray(h_string_ids)
    d_in_values = cp.asarray(h_frequencies)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)
    d_out_num_selected = cp.empty(1, np.int32)

    # Run unique_by_key with automatic temp storage allocation
    parallel.unique_by_key(
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        d_out_num_selected,
        parallel.OpKind.EQUAL_TO,
        d_in_keys.size,
    )

    # Check the result
    num_selected = cp.asnumpy(d_out_num_selected)[0]
    h_out_keys = cp.asnumpy(d_out_keys)[:num_selected]
    h_out_values = cp.asnumpy(d_out_values)[:num_selected]

    # Expected: [1, 2, 3, 4] with first occurrence values [5, 8, 2, 1]
    expected_keys = np.array([1, 2, 3, 4])
    expected_values = np.array([5, 8, 2, 1])

    assert np.array_equal(h_out_keys, expected_keys)
    assert np.array_equal(h_out_values, expected_values)
    print("String deduplication:")
    print(f"Original string IDs: {h_string_ids}")
    print(f"Unique string IDs: {h_out_keys}")
    print(f"Original frequencies: {h_frequencies}")
    print(f"Unique frequencies: {h_out_values}")
    return h_out_keys, h_out_values, num_selected


def keys_only_unique_example():
    """Demonstrate unique by key with keys only (no values)."""

    h_in_keys = np.array([1, 1, 1, 2, 2, 3, 4, 4, 4, 4], dtype="int32")
    d_in_keys = cp.asarray(h_in_keys)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_num_selected = cp.empty(1, np.int32)

    # Run unique_by_key with automatic temp storage allocation
    parallel.unique_by_key(
        d_in_keys,
        None,
        d_out_keys,
        None,
        d_out_num_selected,
        parallel.OpKind.EQUAL_TO,
        d_in_keys.size,
    )

    # Check the result
    num_selected = cp.asnumpy(d_out_num_selected)[0]
    h_out_keys = cp.asnumpy(d_out_keys)[:num_selected]

    # Expected: [1, 2, 3, 4]
    expected_keys = np.array([1, 2, 3, 4])

    assert np.array_equal(h_out_keys, expected_keys)
    print("Keys only unique:")
    print(f"Original keys: {h_in_keys}")
    print(f"Unique keys: {h_out_keys}")
    print(f"Number of unique keys: {num_selected}")
    return h_out_keys, num_selected


if __name__ == "__main__":
    print("Running unique by key examples...")
    basic_unique_by_key_example()
    string_deduplication_example()
    keys_only_unique_example()
    print("All unique by key examples completed successfully!")

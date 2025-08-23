# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Radix sort examples demonstrating the radix sort algorithm.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def basic_radix_sort_example():
    """Demonstrate basic radix sort with keys and values."""
    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)

    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)

    # Run radix_sort with automatic temp storage allocation
    parallel.radix_sort(
        d_in_keys,
        d_out_keys,
        d_in_values,
        d_out_values,
        parallel.SortOrder.ASCENDING,
        d_in_keys.size,
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_out_keys)
    h_out_values = cp.asnumpy(d_out_values)

    argsort = np.argsort(h_in_keys, stable=True)
    expected_keys = np.array(h_in_keys)[argsort]
    expected_values = np.array(h_in_values)[argsort]

    assert np.array_equal(h_out_keys, expected_keys)
    assert np.array_equal(h_out_values, expected_values)
    print(f"Sorted keys: {h_out_keys}")
    print(f"Sorted values: {h_out_values}")
    return h_out_keys, h_out_values


def keys_only_radix_sort_example():
    """Demonstrate radix sort with keys only (no values)."""
    h_in_keys = np.array([64, 8, 32, 16, 4, 2, 1], dtype="int32")
    d_in_keys = cp.asarray(h_in_keys)
    d_out_keys = cp.empty_like(d_in_keys)

    # Run radix_sort with automatic temp storage allocation
    parallel.radix_sort(
        d_in_keys, d_out_keys, None, None, parallel.SortOrder.ASCENDING, d_in_keys.size
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_out_keys)
    expected_keys = np.sort(h_in_keys)

    assert np.array_equal(h_out_keys, expected_keys)
    print(f"Keys only sort result: {h_out_keys}")
    return h_out_keys


def descending_radix_sort_example():
    """Demonstrate radix sort in descending order."""
    h_in_keys = np.array([1, 5, 3, 9, 2, 8, 4, 7, 6], dtype="int32")
    d_in_keys = cp.asarray(h_in_keys)
    d_out_keys = cp.empty_like(d_in_keys)

    # Run radix_sort with automatic temp storage allocation
    parallel.radix_sort(
        d_in_keys, d_out_keys, None, None, parallel.SortOrder.DESCENDING, d_in_keys.size
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_out_keys)
    expected_keys = np.sort(h_in_keys)[::-1]  # Reverse for descending

    assert np.array_equal(h_out_keys, expected_keys)
    print(f"Descending sort result: {h_out_keys}")
    return h_out_keys


if __name__ == "__main__":
    print("Running radix sort examples...")
    basic_radix_sort_example()
    keys_only_radix_sort_example()
    descending_radix_sort_example()
    print("All radix sort examples completed successfully!")

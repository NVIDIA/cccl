# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Merge sort example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def merge_sort_object_example():
    dtype = np.int32
    h_input_keys = np.array([4, 2, 3, 1], dtype=dtype)
    h_input_values = np.array([40, 20, 30, 10], dtype=dtype)
    d_input_keys = cp.asarray(h_input_keys)
    d_input_values = cp.asarray(h_input_values)
    d_output_keys = cp.empty_like(d_input_keys)
    d_output_values = cp.empty_like(d_input_values)

    sorter = parallel.make_merge_sort(
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        parallel.OpKind.LESS,
    )

    temp_storage_size = sorter(
        None,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        len(h_input_keys),
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    sorter(
        d_temp_storage,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        len(h_input_keys),
    )

    expected_keys = np.array([1, 2, 3, 4], dtype=dtype)
    expected_values = np.array([10, 20, 30, 40], dtype=dtype)
    actual_keys = d_output_keys.get()
    actual_values = d_output_values.get()
    print(f"Merge sort object API result keys: {actual_keys}")
    print(f"Merge sort object API result values: {actual_values}")
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_array_equal(actual_values, expected_values)
    print("Merge sort object API example passed.")


if __name__ == "__main__":
    print("Running merge_sort_object_example...")
    merge_sort_object_example()
    print("All examples completed successfully!")

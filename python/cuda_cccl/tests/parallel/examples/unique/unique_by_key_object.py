# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unique by key example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def unique_by_key_object_example():
    dtype = np.int32
    h_input_keys = np.array([1, 1, 2, 3, 3], dtype=dtype)
    h_input_values = np.array([10, 20, 30, 40, 50], dtype=dtype)
    d_input_keys = cp.asarray(h_input_keys)
    d_input_values = cp.asarray(h_input_values)
    d_output_keys = cp.empty_like(d_input_keys)
    d_output_values = cp.empty_like(d_input_values)
    d_num_selected = cp.empty(1, dtype=np.int32)

    uniquer = parallel.make_unique_by_key(
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        d_num_selected,
        parallel.OpKind.EQUAL_TO,
    )

    temp_storage_size = uniquer(
        None,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        d_num_selected,
        len(h_input_keys),
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    uniquer(
        d_temp_storage,
        d_input_keys,
        d_input_values,
        d_output_keys,
        d_output_values,
        d_num_selected,
        len(h_input_keys),
    )

    num_selected = d_num_selected.get()[0]
    expected_keys = np.array([1, 2, 3], dtype=dtype)
    expected_values = np.array([10, 30, 40], dtype=dtype)
    actual_keys = d_output_keys.get()[:num_selected]
    actual_values = d_output_values.get()[:num_selected]
    print(f"Unique by key object API result keys: {actual_keys}")
    print(f"Unique by key object API result values: {actual_values}")
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_array_equal(actual_values, expected_values)
    print("Unique by key object API example passed.")


if __name__ == "__main__":
    print("Running unique_by_key_object_example...")
    unique_by_key_object_example()
    print("All examples completed successfully!")

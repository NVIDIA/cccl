# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Segmented reduction example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def segmented_reduce_object_example():
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    h_input = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty(2, dtype=dtype)

    start_offsets = cp.array([0, 3], dtype=np.int64)
    end_offsets = cp.array([3, 6], dtype=np.int64)

    reducer = parallel.make_segmented_reduce(
        d_input, d_output, start_offsets, end_offsets, parallel.OpKind.PLUS, h_init
    )

    temp_storage_size = reducer(
        None, d_input, d_output, 2, start_offsets, end_offsets, h_init
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    reducer(d_temp_storage, d_input, d_output, 2, start_offsets, end_offsets, h_init)

    expected_result = np.array([6, 15], dtype=dtype)
    actual_result = d_output.get()
    print(f"Segmented reduce object API result: {actual_result}")
    np.testing.assert_array_equal(actual_result, expected_result)
    print("Segmented reduce object API example passed.")


if __name__ == "__main__":
    print("Running segmented_reduce_object_example...")
    segmented_reduce_object_example()
    print("All examples completed successfully!")

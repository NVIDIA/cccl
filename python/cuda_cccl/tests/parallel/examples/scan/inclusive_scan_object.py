# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Inclusive scan example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def inclusive_scan_object_example():
    def add_op(x, y):
        return x + y

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty(len(h_input), dtype=dtype)

    scanner = parallel.make_inclusive_scan(d_input, d_output, add_op, h_init)
    temp_storage_size = scanner.get_temp_storage_bytes(d_input, d_output, len(h_input), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    scanner.compute(d_temp_storage, d_input, d_output, len(h_input), h_init)

    expected_result = np.array([1, 3, 6, 10], dtype=dtype)
    actual_result = d_output.get()
    print(f"Inclusive scan object API result: {actual_result}")
    np.testing.assert_array_equal(actual_result, expected_result)
    print("Inclusive scan object API example passed.")


if __name__ == "__main__":
    print("Running inclusive_scan_object_example...")
    inclusive_scan_object_example()
    print("All examples completed successfully!")

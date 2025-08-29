# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Binary transform examples demonstrating the transform object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def binary_transform_object_example():
    dtype = np.int32
    h_input1 = np.array([1, 2, 3, 4], dtype=dtype)
    h_input2 = np.array([10, 20, 30, 40], dtype=dtype)
    d_input1 = cp.asarray(h_input1)
    d_input2 = cp.asarray(h_input2)
    d_output = cp.empty_like(d_input1)

    transformer = parallel.make_binary_transform(
        d_input1, d_input2, d_output, parallel.OpKind.PLUS
    )

    transformer(d_input1, d_input2, d_output, len(h_input1))

    expected_result = np.array([11, 22, 33, 44], dtype=dtype)
    actual_result = d_output.get()
    print(f"Binary transform object API result: {actual_result}")
    np.testing.assert_array_equal(actual_result, expected_result)
    print("Binary transform object API example passed.")


if __name__ == "__main__":
    print("Running binary transform examples...")
    binary_transform_object_example()
    print("All binary transform examples completed successfully!")

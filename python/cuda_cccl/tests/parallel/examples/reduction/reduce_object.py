# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Reduction example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def reduce_object_example():
    dtype = np.int32
    init_value = 5
    h_init = np.array([init_value], dtype=dtype)
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty(1, dtype=dtype)

    reducer = parallel.make_reduce_into(d_input, d_output, parallel.OpKind.PLUS, h_init)
    temp_storage_size = reducer(None, d_input, d_output, len(h_input), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    reducer(d_temp_storage, d_input, d_output, len(h_input), h_init)

    expected_result = np.sum(h_input) + init_value
    actual_result = d_output.get()[0]
    print(f"Reduce object API result: {actual_result}")
    assert actual_result == expected_result
    print("Reduce object API example passed.")


if __name__ == "__main__":
    print("Running reduce_object_example...")
    reduce_object_example()
    print("All examples completed successfully!")

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
    def add_op(x, y):
        return x + y

    dtype = np.int32
    init_value = 5
    h_init = np.array([init_value], dtype=dtype)
    h_input = np.array([1, 2, 3, 4], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_output = cp.empty(1, dtype=dtype)

    d_input_ptr = d_input.data.ptr
    d_output_ptr = d_output.data.ptr
    h_init_ptr = h_init.data.cast("B")

    reducer = parallel.make_reduce_into(d_input, d_output, add_op, h_init)
    temp_storage_size = reducer(
        0, 0, d_input_ptr, d_output_ptr, len(h_input), h_init_ptr
    )
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    reducer(
        d_temp_storage.data.ptr,
        d_temp_storage.nbytes,
        d_input_ptr,
        d_output_ptr,
        len(h_input),
        h_init_ptr,
    )

    expected_result = np.sum(h_input) + init_value
    actual_result = d_output.get()[0]
    print(f"Reduce object API result: {actual_result}")
    assert actual_result == expected_result
    print("Reduce object API example passed.")


if __name__ == "__main__":
    print("Running reduce_object_example...")
    reduce_object_example()
    print("All examples completed successfully!")

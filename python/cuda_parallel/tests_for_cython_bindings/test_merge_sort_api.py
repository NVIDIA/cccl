# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_merge_sort():
    # example-begin merge-sort
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms._cy_merge_sort as algorithms

    def compare_op(lhs, rhs):
        return np.uint8(lhs < rhs)

    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_items = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )

    d_in_keys = cp.asarray(h_in_keys)
    d_in_items = cp.asarray(h_in_items)

    # Instantiate merge_sort for the given keys, items, and operator
    merge_sort = algorithms.merge_sort(
        d_in_keys, d_in_items, d_in_keys, d_in_items, compare_op
    )

    # Determine temporary device storage requirements
    temp_storage_size = merge_sort(
        None, d_in_keys, d_in_items, d_in_keys, d_in_items, d_in_keys.size
    )

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run merge_sort
    merge_sort(
        d_temp_storage, d_in_keys, d_in_items, d_in_keys, d_in_items, d_in_keys.size
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_in_keys)
    h_out_items = cp.asnumpy(d_in_items)

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_items = np.array(h_in_items)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_items)
    # example-end merge-sort

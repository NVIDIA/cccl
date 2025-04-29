# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_radix_sort():
    # example-begin radix-sort
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms

    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)

    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)

    # Instantiate radix_sort for the given keys, items, and operator
    radix_sort = algorithms.radix_sort(
        d_in_keys, d_out_keys, d_in_values, d_out_values, algorithms.SortOrder.ASCENDING
    )

    # Determine temporary device storage requirements
    temp_storage_size = radix_sort(
        None, d_in_keys, d_out_keys, d_in_values, d_out_values, d_in_keys.size
    )

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run radix_sort
    radix_sort(
        d_temp_storage, d_in_keys, d_out_keys, d_in_values, d_out_values, d_in_keys.size
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(d_out_keys)
    h_out_items = cp.asnumpy(d_out_values)

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_values = np.array(h_in_values)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_values)
    # example-end radix-sort


def test_radix_sort_double_buffer():
    # example-begin radix-sort-buffer
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms

    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)

    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)

    keys_double_buffer = algorithms.DoubleBuffer(d_in_keys, d_out_keys)
    values_double_buffer = algorithms.DoubleBuffer(d_in_values, d_out_values)

    # Instantiate radix_sort for the given keys, items, and operator
    radix_sort = algorithms.radix_sort(
        keys_double_buffer,
        None,
        values_double_buffer,
        None,
        algorithms.SortOrder.ASCENDING,
    )

    # Determine temporary device storage requirements
    temp_storage_size = radix_sort(
        None, keys_double_buffer, None, values_double_buffer, None, d_in_keys.size
    )

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run radix_sort
    radix_sort(
        d_temp_storage,
        keys_double_buffer,
        None,
        values_double_buffer,
        None,
        d_in_keys.size,
    )

    # Check the result is correct
    h_out_keys = cp.asnumpy(keys_double_buffer.current())
    h_out_values = cp.asnumpy(values_double_buffer.current())

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_values = np.array(h_in_values)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_values, h_in_values)
    # example-end radix-sort-buffer

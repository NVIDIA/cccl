# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_unique_by_key():
    # example-begin unique-by-key
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental.algorithms as algorithms

    def compare_op(lhs, rhs):
        return np.uint8(lhs == rhs)

    h_in_keys = np.array([0, 2, 2, 9, 5, 5, 5, 8], dtype="int32")
    h_in_items = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="float32")

    d_in_keys = cp.asarray(h_in_keys)
    d_in_items = cp.asarray(h_in_items)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_items = cp.empty_like(d_in_items)
    d_out_num_selected = cp.empty(1, np.int32)

    # Instantiate unique_by_key for the given keys, items, num items selected, and operator
    unique_by_key = algorithms.unique_by_key(
        d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, compare_op
    )

    # Determine temporary device storage requirements
    temp_storage_size = unique_by_key(
        None,
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        d_in_keys.size,
    )

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run unique_by_key
    unique_by_key(
        d_temp_storage,
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        d_in_keys.size,
    )

    # Check the result is correct
    num_selected = cp.asnumpy(d_out_num_selected)[0]
    h_out_keys = cp.asnumpy(d_out_keys)[:num_selected]
    h_out_items = cp.asnumpy(d_out_items)[:num_selected]

    prev_key = h_in_keys[0]
    expected_keys = [prev_key]
    expected_items = [h_in_items[0]]

    for idx, (previous, next) in enumerate(zip(h_in_keys, h_in_keys[1:])):
        if previous != next:
            expected_keys.append(next)

            # add 1 since we are enumerating over pairs
            expected_items.append(h_in_items[idx + 1])

    np.testing.assert_array_equal(h_out_keys, np.array(expected_keys))
    np.testing.assert_array_equal(h_out_items, np.array(expected_items))
    # example-end unique-by-key

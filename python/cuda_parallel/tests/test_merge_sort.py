# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba.cuda
import numba.types
import numpy as np
import pytest
from conftest import random_array, type_to_problem_sizes

import cuda.parallel.experimental.algorithms as algorithms


def merge_sort_device(
    d_in_keys, d_in_items, d_out_keys, d_out_items, op, num_items, stream=None
):
    merge_sort = algorithms.merge_sort(
        d_in_keys, d_in_items, d_out_keys, d_out_items, op
    )

    temp_storage_size = merge_sort(
        None, d_in_keys, d_in_items, d_out_keys, d_out_items, num_items
    )
    d_temp_storage = numba.cuda.device_array(
        temp_storage_size, dtype=np.uint8, stream=stream.ptr if stream else 0
    )
    merge_sort(
        d_temp_storage, d_in_keys, d_in_items, d_out_keys, d_out_items, num_items
    )


def compare_op(a, b):
    return np.uint8(a < b)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_device_merge_sort_keys(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2

        h_in_keys = random_array(num_items, dtype)
        d_in_keys = numba.cuda.to_device(h_in_keys)

        merge_sort_device(d_in_keys, None, d_in_keys, None, compare_op, num_items)

        h_out_keys = d_in_keys.copy_to_host()
        h_in_keys.sort()

        np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_device_merge_sort_pairs(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2

        h_in_keys = random_array(num_items, dtype)
        h_in_items = random_array(num_items, np.float32)
        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_in_items = numba.cuda.to_device(h_in_items)

        merge_sort_device(
            d_in_keys, d_in_items, d_in_keys, d_in_items, compare_op, num_items
        )

        h_out_keys = d_in_keys.copy_to_host()
        h_out_items = d_in_items.copy_to_host()

        argsort = np.argsort(h_in_keys, stable=True)
        h_in_keys = np.array(h_in_keys)[argsort]
        h_in_items = np.array(h_in_items)[argsort]

        np.testing.assert_array_equal(h_out_keys, h_in_keys)
        np.testing.assert_array_equal(h_out_items, h_in_items)

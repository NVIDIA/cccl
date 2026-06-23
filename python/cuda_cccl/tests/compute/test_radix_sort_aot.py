# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for radix_sort serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import SortOrder, make_radix_sort
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer
from cuda.compute.algorithms._sort._radix_sort import _RadixSort


def _run(sorter, *, d_in_keys, d_out_keys, d_in_values, d_out_values, num_items):
    bytes_needed = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        num_items=num_items,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    sorter(
        temp_storage=tmp,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        num_items=num_items,
    )


def test_serialize_deserialize_radix_sort_keys_values():
    h_in_keys = np.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    h_in_values = np.array(
        [-3.2, 2.2, 1.9, 4.0, -3.9, 2.7, 0, 8.3 - 1, 2.9, 5.4], dtype="float32"
    )
    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)

    builder = make_radix_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        order=SortOrder.ASCENDING,
    )
    blob = builder.serialize()
    assert len(blob) > 0

    loaded = _RadixSort.deserialize(
        blob,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
    )
    _run(
        loaded,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        num_items=d_in_keys.size,
    )

    argsort = np.argsort(h_in_keys, stable=True)
    np.testing.assert_array_equal(d_out_keys.get(), h_in_keys[argsort])
    np.testing.assert_array_equal(d_out_values.get(), h_in_values[argsort])

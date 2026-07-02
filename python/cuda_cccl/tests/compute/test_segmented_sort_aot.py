# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for segmented_sort serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import SortOrder, deserialize, make_segmented_sort, serialize
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

import pytest

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)


def _run(
    sorter,
    *,
    d_in_keys,
    d_out_keys,
    d_in_values,
    d_out_values,
    num_items,
    num_segments,
    start,
    end,
):
    bytes_needed = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=start,
        end_offsets_in=end,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    sorter(
        temp_storage=tmp,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=start,
        end_offsets_in=end,
    )


def test_serialize_deserialize_segmented_sort_round_trip():
    h_in_keys = np.array([9, 1, 5, 4, 2, 8, 7, 3, 6], dtype="int32")
    h_in_vals = np.array([90, 10, 50, 40, 20, 80, 70, 30, 60], dtype="int32")
    start_offsets = np.array([0, 3, 5], dtype=np.int64)
    end_offsets = np.array([3, 5, 9], dtype=np.int64)

    d_in_keys = cp.asarray(h_in_keys)
    d_in_vals = cp.asarray(h_in_vals)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_vals = cp.empty_like(d_in_vals)
    start = cp.asarray(start_offsets)
    end = cp.asarray(end_offsets)

    builder = make_segmented_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_vals,
        d_out_values=d_out_vals,
        start_offsets_in=start,
        end_offsets_in=end,
        order=SortOrder.ASCENDING,
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_vals,
        d_out_values=d_out_vals,
        num_items=d_in_keys.size,
        num_segments=start_offsets.size,
        start=start,
        end=end,
    )

    expected_pairs = []
    for s, e in zip(start_offsets, end_offsets):
        expected_pairs.extend(
            sorted(zip(h_in_keys[s:e], h_in_vals[s:e]), key=lambda kv: kv[0])
        )
    expected_keys = np.array([k for k, _ in expected_pairs], dtype=h_in_keys.dtype)
    expected_vals = np.array([v for _, v in expected_pairs], dtype=h_in_vals.dtype)
    np.testing.assert_array_equal(d_out_keys.get(), expected_keys)
    np.testing.assert_array_equal(d_out_vals.get(), expected_vals)

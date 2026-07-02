# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for unique_by_key serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import OpKind, deserialize, make_unique_by_key, serialize
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
    uniquer,
    *,
    d_in_keys,
    d_in_values,
    d_out_keys,
    d_out_values,
    d_out_num_selected,
    num_items,
    op,
):
    bytes_needed = uniquer(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        d_out_num_selected=d_out_num_selected,
        op=op,
        num_items=num_items,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    uniquer(
        temp_storage=tmp,
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        d_out_num_selected=d_out_num_selected,
        op=op,
        num_items=num_items,
    )


def test_serialize_deserialize_unique_by_key_round_trip():
    h_in_keys = np.array([0, 2, 2, 9, 5, 5, 5, 8], dtype="int32")
    h_in_values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="float32")

    d_in_keys = cp.asarray(h_in_keys)
    d_in_values = cp.asarray(h_in_values)
    d_out_keys = cp.empty_like(d_in_keys)
    d_out_values = cp.empty_like(d_in_values)
    d_out_num_selected = cp.empty(1, np.int32)

    builder = make_unique_by_key(
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        d_out_num_selected=d_out_num_selected,
        op=OpKind.EQUAL_TO,
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    _run(
        loaded,
        d_in_keys=d_in_keys,
        d_in_values=d_in_values,
        d_out_keys=d_out_keys,
        d_out_values=d_out_values,
        d_out_num_selected=d_out_num_selected,
        num_items=d_in_keys.size,
        op=OpKind.EQUAL_TO,
    )

    n = int(d_out_num_selected.get()[0])
    np.testing.assert_array_equal(d_out_keys.get()[:n], np.array([0, 2, 9, 5, 8]))
    np.testing.assert_array_equal(
        d_out_values.get()[:n], np.array([1, 2, 4, 5, 8], dtype=np.float32)
    )

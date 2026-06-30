# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for three_way_partition serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import make_three_way_partition
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer
from cuda.compute.algorithms._three_way_partition import _ThreeWayPartition

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

import pytest

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)


def _less_than_8(x):
    return x < 8 and x >= 0


def _greater_eq_8(x):
    return x >= 8


def _run(
    partitioner,
    *,
    d_in,
    d_first,
    d_second,
    d_unselected,
    d_num_selected,
    num_items,
    op1,
    op2,
):
    bytes_needed = partitioner(
        temp_storage=None,
        d_in=d_in,
        d_first_part_out=d_first,
        d_second_part_out=d_second,
        d_unselected_out=d_unselected,
        d_num_selected_out=d_num_selected,
        select_first_part_op=op1,
        select_second_part_op=op2,
        num_items=num_items,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    partitioner(
        temp_storage=tmp,
        d_in=d_in,
        d_first_part_out=d_first,
        d_second_part_out=d_second,
        d_unselected_out=d_unselected,
        d_num_selected_out=d_num_selected,
        select_first_part_op=op1,
        select_second_part_op=op2,
        num_items=num_items,
    )


def test_serialize_deserialize_three_way_partition_round_trip():
    dtype = np.int32
    h_input = np.array([0, 2, 9, 1, 5, 6, 7, -3, 17, 10], dtype=dtype)
    d_input = cp.asarray(h_input)
    d_first = cp.empty_like(d_input)
    d_second = cp.empty_like(d_input)
    d_unselected = cp.empty_like(d_input)
    d_num_selected = cp.empty(2, dtype=np.int64)

    builder = make_three_way_partition(
        d_in=d_input,
        d_first_part_out=d_first,
        d_second_part_out=d_second,
        d_unselected_out=d_unselected,
        d_num_selected_out=d_num_selected,
        select_first_part_op=_less_than_8,
        select_second_part_op=_greater_eq_8,
    )
    blob = builder.serialize()
    assert len(blob) > 0

    loaded = _ThreeWayPartition.deserialize(blob)
    _run(
        loaded,
        d_in=d_input,
        d_first=d_first,
        d_second=d_second,
        d_unselected=d_unselected,
        d_num_selected=d_num_selected,
        num_items=h_input.size,
        op1=_less_than_8,
        op2=_greater_eq_8,
    )

    actual_num_selected = d_num_selected.get()
    n_first = int(actual_num_selected[0])
    n_second = int(actual_num_selected[1])
    n_unselected = h_input.size - n_first - n_second

    np.testing.assert_array_equal(
        d_first.get()[:n_first], np.array([0, 2, 1, 5, 6, 7], dtype=dtype)
    )
    np.testing.assert_array_equal(
        d_second.get()[:n_second], np.array([9, 17, 10], dtype=dtype)
    )
    np.testing.assert_array_equal(
        d_unselected.get()[:n_unselected], np.array([-3], dtype=dtype)
    )

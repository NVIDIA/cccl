# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for lower_bound / upper_bound serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import make_lower_bound, make_upper_bound, serialize
from cuda.compute.algorithms import load_lower_bound, load_upper_bound

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

import pytest

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)


def test_serialize_deserialize_lower_bound_round_trip():
    h_data = np.array([1, 3, 3, 5, 7, 9], dtype=np.int32)
    h_values = np.array([0, 3, 4, 10], dtype=np.int32)
    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(len(h_values), dtype=np.uintp)

    builder = make_lower_bound(d_data=d_data, d_values=d_values, d_out=d_out)
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = load_lower_bound(blob)
    loaded(
        d_data=d_data,
        num_items=len(d_data),
        d_values=d_values,
        num_values=len(d_values),
        d_out=d_out,
        comp=None,
    )

    expected = np.searchsorted(h_data, h_values, side="left").astype(np.uintp)
    np.testing.assert_array_equal(d_out.get(), expected)


def test_serialize_deserialize_upper_bound_round_trip():
    h_data = np.array([1, 3, 3, 5, 7, 9], dtype=np.int32)
    h_values = np.array([0, 3, 4, 10], dtype=np.int32)
    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(len(h_values), dtype=np.uintp)

    builder = make_upper_bound(d_data=d_data, d_values=d_values, d_out=d_out)
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = load_upper_bound(blob)
    loaded(
        d_data=d_data,
        num_items=len(d_data),
        d_values=d_values,
        num_values=len(d_values),
        d_out=d_out,
        comp=None,
    )

    expected = np.searchsorted(h_data, h_values, side="right").astype(np.uintp)
    np.testing.assert_array_equal(d_out.get(), expected)

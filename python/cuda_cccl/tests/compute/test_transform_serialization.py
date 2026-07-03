# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for unary/binary transform serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import (
    OpKind,
    deserialize,
    make_binary_transform,
    make_unary_transform,
    serialize,
)

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

import pytest

pytestmark = pytest.mark.skipif(
    USING_V2, reason="serialization not supported on v2 (HostJIT) backend"
)


def _add_one(a):
    return a + 1


def test_serialize_deserialize_unary_transform_round_trip():
    h_in = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)

    builder = make_unary_transform(d_in=d_in, d_out=d_out, op=_add_one)
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    loaded(d_in=d_in, d_out=d_out, op=_add_one, num_items=d_in.size)

    np.testing.assert_array_equal(d_out.get(), h_in + 1)


def test_serialize_deserialize_binary_transform_round_trip():
    h_in1 = np.array([1, 2, 3, 4], dtype=np.int32)
    h_in2 = np.array([10, 20, 30, 40], dtype=np.int32)
    d_in1 = cp.asarray(h_in1)
    d_in2 = cp.asarray(h_in2)
    d_out = cp.empty_like(d_in1)

    builder = make_binary_transform(
        d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=OpKind.PLUS
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    loaded(d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=OpKind.PLUS, num_items=d_in1.size)

    np.testing.assert_array_equal(d_out.get(), h_in1 + h_in2)

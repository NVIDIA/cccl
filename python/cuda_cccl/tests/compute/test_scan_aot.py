# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for scan serialize/deserialize AoT APIs."""

import cupy as cp
import numpy as np

from cuda.compute import (
    OpKind,
    exclusive_scan,
    make_exclusive_scan,
    make_inclusive_scan,
)
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer
from cuda.compute.algorithms._scan import _Scan

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

import pytest

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)


def _run(scanner, *, d_in, d_out, op, init_value, num_items):
    bytes_needed = scanner(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        op=op,
        init_value=init_value,
        num_items=num_items,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    scanner(
        temp_storage=tmp,
        d_in=d_in,
        d_out=d_out,
        op=op,
        init_value=init_value,
        num_items=num_items,
    )


def test_serialize_deserialize_exclusive_scan_round_trip():
    d_in = cp.arange(1, 33, dtype=cp.int32)
    d_out = cp.empty_like(d_in)
    init_value = np.array([0], dtype=np.int32)

    builder = make_exclusive_scan(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=init_value
    )
    blob = builder.serialize()
    assert len(blob) > 0

    loaded = _Scan.deserialize(
        blob,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        init_value=init_value,
        force_inclusive=False,
    )
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        init_value=init_value,
        num_items=d_in.size,
    )

    expected = np.zeros_like(d_in.get())
    np.cumsum(d_in.get()[:-1], out=expected[1:])
    np.testing.assert_array_equal(d_out.get(), expected)


def test_serialize_deserialize_inclusive_scan_round_trip():
    d_in = cp.arange(1, 33, dtype=cp.int32)
    d_out = cp.empty_like(d_in)
    init_value = np.array([0], dtype=np.int32)

    builder = make_inclusive_scan(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=init_value
    )
    blob = builder.serialize()

    loaded = _Scan.deserialize(
        blob,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        init_value=init_value,
        force_inclusive=True,
    )
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        init_value=init_value,
        num_items=d_in.size,
    )

    np.testing.assert_array_equal(d_out.get(), np.cumsum(d_in.get()))


def test_deserialize_after_jit_matches_jit_result():
    """Serialize a JITed scan, deserialize, and confirm output matches a fresh JIT."""
    d_in = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype=cp.int32)
    d_out_jit = cp.empty_like(d_in)
    d_out_aot = cp.empty_like(d_in)
    init_value = np.array([1], dtype=np.int32)

    def max_op(a, b):
        return a if a > b else b

    exclusive_scan(
        d_in=d_in,
        d_out=d_out_jit,
        op=max_op,
        init_value=init_value,
        num_items=d_in.size,
    )

    builder = make_exclusive_scan(
        d_in=d_in, d_out=d_out_aot, op=max_op, init_value=init_value
    )
    blob = builder.serialize()
    loaded = _Scan.deserialize(
        blob,
        d_in=d_in,
        d_out=d_out_aot,
        op=max_op,
        init_value=init_value,
        force_inclusive=False,
    )
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out_aot,
        op=max_op,
        init_value=init_value,
        num_items=d_in.size,
    )

    np.testing.assert_array_equal(d_out_aot.get(), d_out_jit.get())

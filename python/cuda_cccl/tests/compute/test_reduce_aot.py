# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the ahead-of-time serialize/deserialize reduce Python API."""

import cupy as cp
import numpy as np
import pytest

from cuda.compute import (
    Determinism,
    OpKind,
    make_reduce_into,
)
from cuda.compute.algorithms._reduce import _Reduce

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)


def _add(a, b):
    return a + b


def test_serialize_deserialize_well_known_op_round_trip():
    d_in = cp.arange(1024, dtype=cp.int32)
    d_out = cp.zeros(1, dtype=cp.int32)
    h_init = np.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init)
    blob = reducer.serialize()
    assert len(blob) > 0

    loaded = _Reduce.deserialize(
        blob, d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )

    # Loaded reducer is fully usable without any JIT.
    from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

    bytes_needed = loaded(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_items=d_in.size,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    loaded(
        temp_storage=tmp,
        d_in=d_in,
        d_out=d_out,
        num_items=d_in.size,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    expected = int(cp.sum(d_in).get())
    assert int(d_out[0].get()) == expected


def test_serialize_deserialize_jit_op_round_trip():
    d_in = cp.arange(1024, dtype=cp.int32)
    d_out = cp.zeros(1, dtype=cp.int32)
    h_init = np.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=_add, h_init=h_init)
    blob = reducer.serialize()

    loaded = _Reduce.deserialize(blob, d_in=d_in, d_out=d_out, op=_add, h_init=h_init)

    from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

    bytes_needed = loaded(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_items=d_in.size,
        op=_add,
        h_init=h_init,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    loaded(
        temp_storage=tmp,
        d_in=d_in,
        d_out=d_out,
        num_items=d_in.size,
        op=_add,
        h_init=h_init,
    )

    assert int(d_out[0].get()) == int(cp.sum(d_in).get())


def test_serialize_deserialize_preserves_determinism():
    d_in = cp.arange(64, dtype=cp.int32)
    d_out = cp.zeros(1, dtype=cp.int32)
    h_init = np.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        h_init=h_init,
        determinism=Determinism.NOT_GUARANTEED,
    )
    blob = reducer.serialize()

    loaded = _Reduce.deserialize(
        blob, d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )
    assert loaded.build_result.determinism == int(Determinism.NOT_GUARANTEED)


def test_deserialize_garbage_raises():
    d_in = cp.arange(8, dtype=cp.int32)
    d_out = cp.zeros(1, dtype=cp.int32)
    with pytest.raises(RuntimeError):
        _Reduce.deserialize(
            b"not a real aot blob" + b"\0" * 64,
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            h_init=np.zeros(1, dtype=np.int32),
        )

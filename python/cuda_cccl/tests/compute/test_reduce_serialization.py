# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the serialize/deserialize reduce Python API.

``deserialize`` takes only the blob: the iterators, operator, and init value are
rebuilt from the descriptor sidecar embedded in it. Live device pointers and
operator/init state are bound per call.
"""

import cupy as cp
import numpy as np
import pytest

from cuda.compute import (
    CountingIterator,
    Determinism,
    OpKind,
    TransformIterator,
    deserialize,
    make_reduce_into,
    serialize,
)
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    USING_V2, reason="serialization not supported on v2 (HostJIT) backend"
)


def _add(a, b):
    return a + b


def _plus_one(x):
    return x + 1


def _run(loaded, *, d_in, d_out, num_items, op, h_init):
    """Drive a loaded reducer through the (size query, execute) two-step."""
    bytes_needed = loaded(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=op,
        h_init=h_init,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    loaded(
        temp_storage=tmp,
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=op,
        h_init=h_init,
    )


def test_serialize_deserialize_well_known_op_round_trip():
    d_in = cp.arange(1024, dtype=cp.int32)
    d_out = cp.zeros(1, dtype=cp.int32)
    h_init = np.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init)
    blob = serialize(reducer)
    assert len(blob) > 0

    # Loaded reducer is fully usable without any JIT and without supplying objects.
    loaded = deserialize(blob)
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out,
        num_items=d_in.size,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    assert int(d_out[0].get()) == int(cp.sum(d_in).get())


def test_serialize_deserialize_jit_op_round_trip():
    d_in = cp.arange(1024, dtype=cp.int32)
    d_out = cp.zeros(1, dtype=cp.int32)
    h_init = np.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=_add, h_init=h_init)
    blob = serialize(reducer)

    loaded = deserialize(blob)
    # The user op's device code is rebuilt from the blob, not from a re-supplied
    # / recompiled operator: assert it is present before any object reaches the
    # call below (deserialize took only the blob).
    assert len(loaded.op_cccl.ltoir) > 0
    _run(loaded, d_in=d_in, d_out=d_out, num_items=d_in.size, op=_add, h_init=h_init)

    assert int(d_out[0].get()) == int(cp.sum(d_in).get())


def test_serialize_deserialize_counting_iterator_input():
    # Custom (ITERATOR-kind) input: its device advance/dereference code must be
    # captured in the descriptor sidecar and rebuilt with no object supplied.
    n = 1024
    d_out = cp.zeros(1, dtype=np.int64)
    h_init = np.zeros(1, dtype=np.int64)

    reducer = make_reduce_into(
        d_in=CountingIterator(np.int32(0)), d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )
    blob = serialize(reducer)

    loaded = deserialize(blob)
    # The ITERATOR-kind descriptor (with its advance/dereference device code) was
    # rebuilt purely from the blob — no iterator object was passed to deserialize,
    # so a regression to caller-supplied descriptors would fail here, not silently
    # pass via the object reconstructed for the call.
    assert loaded.d_in_cccl.is_kind_iterator()
    _run(
        loaded,
        d_in=CountingIterator(np.int32(0)),
        d_out=d_out,
        num_items=n,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    assert int(d_out[0].get()) == n * (n - 1) // 2


def test_serialize_deserialize_transform_iterator_input():
    # TransformIterator carries a user op (device code) inside the iterator's
    # dereference op — exercises iterator-embedded LTOIR round-tripping.
    n = 512
    d_out = cp.zeros(1, dtype=np.int64)
    h_init = np.zeros(1, dtype=np.int64)

    def make_it():
        return TransformIterator(CountingIterator(np.int32(0)), _plus_one)

    reducer = make_reduce_into(
        d_in=make_it(), d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )
    blob = serialize(reducer)

    loaded = deserialize(blob)
    # Iterator descriptor (incl. the transform op's embedded LTOIR) rebuilt from
    # the blob alone — deserialize took no objects.
    assert loaded.d_in_cccl.is_kind_iterator()
    _run(
        loaded, d_in=make_it(), d_out=d_out, num_items=n, op=OpKind.PLUS, h_init=h_init
    )

    # sum of (i + 1) for i in 0..n-1
    assert int(d_out[0].get()) == n * (n - 1) // 2 + n


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
    blob = serialize(reducer)

    loaded = deserialize(blob)
    assert loaded.build_result.determinism == int(Determinism.NOT_GUARANTEED)


def test_deserialize_garbage_raises():
    with pytest.raises((ValueError, RuntimeError)):
        deserialize(b"not a real serialization blob" + b"\0" * 64)

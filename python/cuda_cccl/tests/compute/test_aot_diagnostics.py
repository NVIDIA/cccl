# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AoT deserialize diagnostics.

Before these checks, every C-side deserialize/load failure surfaced in Python as
an opaque ``error code: <n>`` with the real reason printed only to stdout. The C
layer now validates the blob header (magic / format / ABI version) and — for
CUBIN payloads — the target compute-capability major against the current device,
*before* the opaque ``cuLibraryLoadData`` failure, and propagates a descriptive
message via ``cccl_aot_last_error()``.

The compute-capability case is exercised single-GPU by patching the ``cc`` field
of a real blob (a true cross-GPU load is the same code path but needs a second
architecture).
"""

import struct

import cupy as cp
import numpy as np
import pytest

from cuda.compute import OpKind, deserialize, make_reduce_into

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)

_C_MAGIC = b"CCCLAOT1"
# Field offsets within the C build_result header, past the 8-byte magic:
#   algo_tag u32 | format_version u32 | cccl_version u64 | payload_kind u32 | cc u32
_OFF_CCCL_VERSION = 4 + 4
_OFF_CC = 4 + 4 + 8 + 4


def _reduce_blob():
    d = cp.arange(1024, dtype=cp.int32)
    o = cp.zeros(1, dtype=cp.int32)
    r = make_reduce_into(d_in=d, d_out=o, op=OpKind.PLUS, h_init=np.zeros(1, np.int32))
    return r.serialize()


def _patch_u32(blob, field_off, value):
    b = bytearray(blob)
    i = b.find(_C_MAGIC)
    struct.pack_into("<I", b, i + len(_C_MAGIC) + field_off, value)
    return bytes(b)


def _patch_u64(blob, field_off, value):
    b = bytearray(blob)
    i = b.find(_C_MAGIC)
    struct.pack_into("<Q", b, i + len(_C_MAGIC) + field_off, value)
    return bytes(b)


def test_abi_mismatch_reports_clear_message():
    bad = _patch_u64(_reduce_blob(), _OFF_CCCL_VERSION, 999)
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        deserialize(bad)


def test_wrong_cc_major_reports_clear_message():
    maj = int(cp.cuda.Device().compute_capability[0])
    other_major = 7 if maj != 7 else 8
    bad = _patch_u32(_reduce_blob(), _OFF_CC, other_major * 10 + 5)
    with pytest.raises(RuntimeError, match="compute-capability major"):
        deserialize(bad)


def test_bad_magic_reports_clear_message():
    b = bytearray(_reduce_blob())
    i = b.find(_C_MAGIC)
    b[i : i + len(_C_MAGIC)] = b"XXXXXXXX"  # corrupt only the C header magic
    with pytest.raises(RuntimeError, match="bad magic"):
        deserialize(bytes(b))


def test_valid_blob_still_roundtrips():
    from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

    reducer = deserialize(_reduce_blob())
    d = cp.arange(1024, dtype=cp.int32)
    o = cp.zeros(1, dtype=cp.int32)
    kw = dict(
        d_in=d, d_out=o, num_items=1024, op=OpKind.PLUS, h_init=np.zeros(1, np.int32)
    )
    nbytes = reducer(temp_storage=None, **kw)
    reducer(temp_storage=TempStorageBuffer(nbytes, None), **kw)
    assert int(o.get()[0]) == 1024 * 1023 // 2

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for serialization deserialize diagnostics.

A deserialize/load failure should carry an actionable message, not an opaque
``error code: <n>``. The C layer validates the blob header magic and, for CUBIN
payloads, the target compute-capability major against the current device before
the opaque ``cuLibraryLoadData`` failure, and propagates a descriptive message
via ``cccl_serialization_last_error()``.

The compute-capability case is exercised single-GPU by patching the ``cc`` field
of a real blob (a true cross-GPU load is the same code path but needs a second
architecture).
"""

import struct

import numpy as np
import pytest
from _utils.device_array import DeviceArray, get_compute_capability

from cuda.compute import OpKind, deserialize, make_reduce_into

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    USING_V2, reason="serialization not supported on v2 (HostJIT) backend"
)

_C_MAGIC = b"CCCLSER1"
# Field offsets within the C build_result header, past the 8-byte magic:
#   algo_tag u32 | payload_kind u32 | cc u32
_OFF_CC = 4 + 4


def _reduce_blob():
    d = DeviceArray.from_numpy(np.arange(1024, dtype=np.int32))
    o = DeviceArray.from_numpy(np.zeros(1, dtype=np.int32))
    r = make_reduce_into(d_in=d, d_out=o, op=OpKind.PLUS, h_init=np.zeros(1, np.int32))
    return r.serialize()


def _patch_u32(blob, field_off, value):
    b = bytearray(blob)
    i = b.find(_C_MAGIC)
    struct.pack_into("<I", b, i + len(_C_MAGIC) + field_off, value)
    return bytes(b)


def test_wrong_cc_major_reports_clear_message():
    maj = int(get_compute_capability()[0])
    other_major = 7 if maj != 7 else 8
    bad = _patch_u32(_reduce_blob(), _OFF_CC, other_major * 10 + 5)
    with pytest.raises(RuntimeError, match="compute-capability major"):
        deserialize(bad)


def test_wrong_cc_minor_reports_clear_message():
    # A CUBIN built for a higher minor than the device is not backward-compatible
    # (SASS is only forward-compatible across minors within a major), so a blob
    # targeting the same major but a higher minor must be rejected.
    maj, minr = (int(part) for part in get_compute_capability())
    if minr >= 9:
        pytest.skip("device minor is already the maximum within its major")
    bad = _patch_u32(_reduce_blob(), _OFF_CC, maj * 10 + (minr + 1))
    with pytest.raises(RuntimeError, match="device minor"):
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
    d = DeviceArray.from_numpy(np.arange(1024, dtype=np.int32))
    o = DeviceArray.from_numpy(np.zeros(1, dtype=np.int32))
    kw = dict(
        d_in=d, d_out=o, num_items=1024, op=OpKind.PLUS, h_init=np.zeros(1, np.int32)
    )
    nbytes = reducer(temp_storage=None, **kw)
    reducer(temp_storage=TempStorageBuffer(nbytes, None), **kw)
    assert int(o.copy_to_host()[0]) == 1024 * 1023 // 2

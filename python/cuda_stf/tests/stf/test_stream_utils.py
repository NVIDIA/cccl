# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for stream-pointer resolution (``__cuda_stream__`` protocol support).

The pure-Python tests below exercise ``get_stream_pointer`` directly and do
not require a GPU or the compiled STF extension. The GPU-gated test verifies
that ``context(stream=...)`` accepts a ``__cuda_stream__`` object end to end.
"""

import pytest

from cuda.stf._experimental._stream_utils import get_stream_pointer


class _FakeStream:
    """Minimal object implementing the ``__cuda_stream__`` protocol."""

    def __init__(self, handle, version=0):
        self._handle = handle
        self._version = version

    def __cuda_stream__(self):
        return (self._version, self._handle)


def test_none_maps_to_null_stream():
    assert get_stream_pointer(None) == 0


def test_plain_int_pointer_is_passed_through():
    assert get_stream_pointer(0) == 0
    assert get_stream_pointer(0xDEADBEEF) == 0xDEADBEEF


def test_cuda_stream_protocol_object():
    assert get_stream_pointer(_FakeStream(0x1234)) == 0x1234


def test_cuda_stream_protocol_takes_precedence_over_int_coercion():
    # An object whose int() coercion would differ from its protocol handle
    # must resolve via the protocol, not via int().
    class _IntLikeStream(int):
        def __cuda_stream__(self):
            return (0, 0x4242)

    obj = _IntLikeStream(999)
    assert get_stream_pointer(obj) == 0x4242


def test_rejects_object_without_protocol_or_int():
    with pytest.raises(TypeError):
        get_stream_pointer(object())


def test_rejects_unsupported_protocol_version():
    with pytest.raises(TypeError):
        get_stream_pointer(_FakeStream(0x1234, version=1))


def test_rejects_non_int_handle():
    with pytest.raises(TypeError):
        get_stream_pointer(_FakeStream("not-an-int"))


def test_rejects_malformed_protocol_return():
    class _BadStream:
        def __cuda_stream__(self):
            return None

    with pytest.raises(TypeError):
        get_stream_pointer(_BadStream())


def test_context_accepts_stream_protocol_object():
    """End-to-end: a ``__cuda_stream__`` object flows into ``context(stream=...)``."""
    pytest.importorskip("cuda.stf._experimental._stf_bindings")
    core = pytest.importorskip("cuda.core.experimental")
    import cuda.stf._experimental as stf

    dev = core.Device()
    dev.set_current()
    stream = dev.create_stream()

    ctx = stf.context(stream=stream)
    ctx.finalize()

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import weakref

import pytest

from cuda.compute._utils import protocols


class _CountingStream:
    def __init__(self, handle):
        self.handle = handle
        self.calls = 0

    def __cuda_stream__(self):
        self.calls += 1
        return (0, self.handle)


def test_validate_and_get_stream_caches_handle_by_identity():
    stream = _CountingStream(123)

    assert protocols.validate_and_get_stream(stream) == 123
    assert protocols.validate_and_get_stream(stream) == 123
    assert stream.calls == 1


def test_validate_and_get_stream_does_not_confuse_equal_objects():
    class EqualStream(_CountingStream):
        def __eq__(self, other):
            return isinstance(other, EqualStream)

        def __hash__(self):
            return 1

    first = EqualStream(123)
    second = EqualStream(456)

    assert first == second
    assert protocols.validate_and_get_stream(first) == 123
    assert protocols.validate_and_get_stream(second) == 456
    assert first.calls == 1
    assert second.calls == 1


def test_stream_handle_cache_does_not_extend_stream_lifetime():
    stream = _CountingStream(123)
    stream_id = id(stream)
    stream_ref = weakref.ref(stream)

    assert protocols.validate_and_get_stream(stream) == 123
    assert stream_id in protocols._STREAM_HANDLE_CACHE

    del stream
    gc.collect()

    assert stream_ref() is None
    assert stream_id not in protocols._STREAM_HANDLE_CACHE


def test_non_weakrefable_stream_remains_supported_without_caching():
    class NonWeakrefableStream:
        __slots__ = ("calls",)

        def __init__(self):
            self.calls = 0

        def __cuda_stream__(self):
            self.calls += 1
            return (0, 123)

    stream = NonWeakrefableStream()

    assert protocols.validate_and_get_stream(stream) == 123
    assert protocols.validate_and_get_stream(stream) == 123
    assert stream.calls == 2


def test_invalid_stream_is_not_cached():
    class InvalidStream(_CountingStream):
        def __cuda_stream__(self):
            self.calls += 1
            return (0, None)

    stream = InvalidStream(0)

    with pytest.raises(TypeError, match="invalid stream handle"):
        protocols.validate_and_get_stream(stream)
    with pytest.raises(TypeError, match="invalid stream handle"):
        protocols.validate_and_get_stream(stream)
    assert stream.calls == 2

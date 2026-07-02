# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
from cuda.stf._experimental import fill_utils


class _FakeTask:
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get_arg_cai(self, index):
        assert index == 0
        return {
            "data": (1234, False),
            "shape": (4,),
            "typestr": self.dtype.str,
        }

    def stream_ptr(self):
        return 5678


class _FakeContext:
    def __init__(self, dtype):
        self.dtype = dtype

    def task(self, *args):
        assert args == ("write-dep",)
        return _FakeTask(self.dtype)


class _FakeLogicalData:
    def write(self):
        return "write-dep"


def test_init_logical_data_uses_cuda_core_for_8_byte_zero_fill(monkeypatch):
    fill_calls = []

    class FakeBuffer:
        @classmethod
        def from_handle(cls, ptr, size, owner=None):
            assert ptr == 1234
            assert size == 4 * np.dtype(np.float64).itemsize
            assert owner is None
            return cls()

        def fill(self, value, *, stream):
            fill_calls.append((value, stream))

    class FakeStream:
        @classmethod
        def from_handle(cls, handle):
            assert handle == 5678
            return "stream"

    def fail_cupy_fallback(*args):
        raise AssertionError("8-byte zero fill should not require CuPy")

    monkeypatch.setattr(fill_utils, "Buffer", FakeBuffer)
    monkeypatch.setattr(fill_utils, "Stream", FakeStream)
    monkeypatch.setattr(fill_utils, "_fill_8byte_cupy", fail_cupy_fallback)

    fill_utils.init_logical_data(_FakeContext(np.float64), _FakeLogicalData(), 0.0)

    assert fill_calls == [(0, "stream")]


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_init_logical_data_still_uses_cupy_for_nonzero_8_byte_fill(monkeypatch, dtype):
    fallback_calls = []

    class FakeBuffer:
        @classmethod
        def from_handle(cls, ptr, size, owner=None):
            return cls()

        def fill(self, value, *, stream):
            raise AssertionError("nonzero 8-byte fill cannot use cuda.core Buffer.fill")

    class FakeStream:
        @classmethod
        def from_handle(cls, handle):
            return "stream"

    def record_cupy_fallback(shape, dtype, value, ptr, size, stream_ptr):
        fallback_calls.append((shape, dtype, value, ptr, size, stream_ptr))

    monkeypatch.setattr(fill_utils, "Buffer", FakeBuffer)
    monkeypatch.setattr(fill_utils, "Stream", FakeStream)
    monkeypatch.setattr(fill_utils, "_fill_8byte_cupy", record_cupy_fallback)

    fill_utils.init_logical_data(_FakeContext(dtype), _FakeLogicalData(), 1)

    expected_size = 4 * np.dtype(dtype).itemsize
    assert fallback_calls == [((4,), np.dtype(dtype), 1, 1234, expected_size, 5678)]

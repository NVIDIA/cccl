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


class _FakeScalarTask(_FakeTask):
    """A task whose argument is a 0-d scalar (empty shape)."""

    def get_arg_cai(self, index):
        assert index == 0
        return {
            "data": (1234, False),
            "shape": (),
            "typestr": self.dtype.str,
        }


class _FakeScalarContext(_FakeContext):
    def task(self, *args):
        assert args == ("write-dep",)
        return _FakeScalarTask(self.dtype)


class _FakeLogicalData:
    def write(self):
        return "write-dep"


def test_init_logical_data_fills_scalar_shape(monkeypatch):
    """A 0-d scalar (shape ()) is one element and must still be filled."""
    fill_calls = []

    class FakeBuffer:
        @classmethod
        def from_handle(cls, ptr, size, owner=None):
            # Empty shape => exactly one element, so size == itemsize (not 0).
            assert size == np.dtype(np.float32).itemsize
            return cls()

        def fill(self, value, *, stream):
            fill_calls.append((value, stream))

    class FakeStream:
        @classmethod
        def from_handle(cls, handle):
            return "stream"

    monkeypatch.setattr(fill_utils, "Buffer", FakeBuffer)
    monkeypatch.setattr(fill_utils, "Stream", FakeStream)

    fill_utils.init_logical_data(_FakeScalarContext(np.float32), _FakeLogicalData(), 7)

    assert len(fill_calls) == 1


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

    def fail_driver_fill(*args):
        raise AssertionError("8-byte zero fill should not require the driver memset")

    monkeypatch.setattr(fill_utils, "Buffer", FakeBuffer)
    monkeypatch.setattr(fill_utils, "Stream", FakeStream)
    monkeypatch.setattr(fill_utils, "_fill_8byte_driver", fail_driver_fill)

    fill_utils.init_logical_data(_FakeContext(np.float64), _FakeLogicalData(), 0.0)

    # A bytewise zero fill is valid for any dtype and goes through cuda.core.
    assert fill_calls == [(0, "stream")]


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_init_logical_data_uses_driver_memset_for_nonzero_8_byte_fill(
    monkeypatch, dtype
):
    driver_calls = []

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

    def record_driver_fill(dtype, value, ptr, count, stream_ptr):
        driver_calls.append((dtype, value, ptr, count, stream_ptr))

    monkeypatch.setattr(fill_utils, "Buffer", FakeBuffer)
    monkeypatch.setattr(fill_utils, "Stream", FakeStream)
    monkeypatch.setattr(fill_utils, "_fill_8byte_driver", record_driver_fill)

    fill_utils.init_logical_data(_FakeContext(dtype), _FakeLogicalData(), 1)

    # Nonzero 8-byte fills use a pair of strided 32-bit driver memsets rather
    # than cuda.core's fill (which only supports 1/2/4-byte patterns).
    assert driver_calls == [(np.dtype(dtype), 1, 1234, 4, 5678)]

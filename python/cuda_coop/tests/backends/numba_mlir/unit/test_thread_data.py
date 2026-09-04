# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import struct

import pytest

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir import _thread_data

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


class _FakeLocal:
    def __init__(self):
        self.calls = []

    def array(self, shape, dtype, *, alignment, **kwargs):
        call = (shape, dtype, alignment, kwargs)
        self.calls.append(call)
        return call


class _FakeRuntime:
    def __init__(self):
        self.local = _FakeLocal()


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = _FakeRuntime()
    monkeypatch.setattr(_thread_data, "_require_runtime", lambda: runtime)
    return runtime


def test_thread_data_accepts_canonical_extent_forms(fake_runtime):
    result = coop.ThreadData(
        items_per_thread=4,
        dtype="int32",
        alignment=16,
    )
    compatible = coop.ThreadData(1, "int16", alignas=16)
    inferred = coop.ThreadData(2)

    assert result == (4, "int32", 16, {})
    assert compatible == (1, "int16", 16, {})
    assert inferred == (2, None, 8, {})
    assert fake_runtime.local.calls == [result, compatible, inferred]


def test_thread_data_rejects_conflicting_alignment_aliases(fake_runtime):
    del fake_runtime
    with pytest.raises(ValueError, match="alignas and alignment must match"):
        coop.ThreadData(1, alignas=16, alignment=32)
    with pytest.raises(ValueError, match="alignas and alignment must match"):
        coop.ThreadData(1, alignas=8, alignment=16)


@pytest.mark.parametrize(
    "call",
    [
        lambda: coop.ThreadData(shape=4),
        lambda: coop.ThreadData(4, address_space="local"),
        lambda: coop.ThreadData(4, items_per_thread=4),
    ],
)
def test_thread_data_rejects_unknown_or_duplicate_arguments(fake_runtime, call):
    del fake_runtime
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize(
    ("items_per_thread", "error_type", "message"),
    [
        (True, TypeError, "items_per_thread must be an integer"),
        (1.5, TypeError, "items_per_thread must be an integer"),
        (0, ValueError, "items_per_thread must be a positive integer"),
        (-1, ValueError, "items_per_thread must be a positive integer"),
    ],
)
def test_thread_data_validates_extent(
    fake_runtime,
    items_per_thread,
    error_type,
    message,
):
    del fake_runtime
    with pytest.raises(error_type, match=message):
        coop.ThreadData(items_per_thread)


@pytest.mark.parametrize(
    ("alignment", "error_type", "message"),
    [
        (True, TypeError, "alignment must be an integer"),
        (1.5, TypeError, "alignment must be an integer"),
        (0, ValueError, "alignment must be a positive integer"),
        (-1, ValueError, "alignment must be a positive integer"),
        (3, ValueError, "alignment must be a power of 2"),
        (
            struct.calcsize("P") // 2,
            ValueError,
            f"alignment must be a multiple of {struct.calcsize('P')}",
        ),
    ],
)
def test_thread_data_validates_alignment(
    fake_runtime,
    alignment,
    error_type,
    message,
):
    del fake_runtime
    with pytest.raises(error_type, match=message):
        coop.ThreadData(1, alignment=alignment)


def test_temp_storage_uses_canonical_defaults_and_normalization():
    shared = coop.TempStorage(sharing=" SHARED ")
    exclusive = coop.TempStorage(sharing=" Exclusive ")

    assert shared.sharing == "shared"
    assert shared.auto_sync is True
    assert exclusive.sharing == "exclusive"
    assert exclusive.auto_sync is False


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"size_in_bytes": True},
            TypeError,
            "TempStorage size_in_bytes must be an integer or None",
        ),
        (
            {"size_in_bytes": 0},
            ValueError,
            "TempStorage size_in_bytes must be a positive integer",
        ),
        (
            {"alignment": False},
            TypeError,
            "TempStorage alignment must be an integer or None",
        ),
        (
            {"alignment": 3},
            ValueError,
            "TempStorage alignment must be a power of 2",
        ),
        (
            {"auto_sync": 1},
            TypeError,
            "TempStorage auto_sync must be None/True/False",
        ),
        (
            {"sharing": 1},
            TypeError,
            "TempStorage sharing must be a string",
        ),
        (
            {"sharing": "exclusive", "auto_sync": True},
            ValueError,
            "sharing='exclusive'.*auto_sync=True",
        ),
    ],
)
def test_temp_storage_validation(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        coop.TempStorage(**kwargs)

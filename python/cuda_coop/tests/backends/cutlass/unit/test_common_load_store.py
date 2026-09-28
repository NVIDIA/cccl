# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common load store frontend checks for CUTLASS tracing."""

from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core import this_block, this_warp

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _ThreadData:
    def __init__(self, items_per_thread=2, *, dtype=np.int32, length=None):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [0] * (items_per_thread if length is None else length)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value


class _ReadonlyThreadData:
    def __init__(self, items_per_thread=2, *, dtype=np.int32):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [0] * items_per_thread

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _TempStorage:
    size_in_bytes = 128
    alignment = 16
    auto_sync = True
    sharing = "shared"


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(this_block(), id="block"),
        pytest.param(this_warp(), id="physical-warp"),
        pytest.param(this_warp().group_by(8), id="logical-warp"),
    ],
)
def test_common_load_mutates_output_and_returns_none(monkeypatch, group):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    output = _ThreadData()
    source = [11, 23]
    calls = []

    def marker(operation, selected_group, selected_source, selected_output, **kwargs):
        calls.append((operation, selected_group, selected_source, selected_output))
        for index, value in enumerate(selected_source):
            selected_output[index] = value

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.load_store"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        assert api.load(group, source, output) is None
    assert calls == [("load", group, source, output)]
    assert list(output) == source


def test_common_load_store_validate_block_payloads_and_options(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    monkeypatch.setattr(
        api, "_group_primitive_marker", lambda operation, *args, **kwargs: None
    )
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.load_store"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        api.store(
            this_block(), object(), _ReadonlyThreadData(), temp_storage=_TempStorage()
        )
        with pytest.raises(ValueError, match="oob_default requires valid_items"):
            api.load(this_block(), object(), _ThreadData(), oob_default=0)
        with pytest.raises(TypeError, match="must satisfy TempStorageLike"):
            api.load(this_block(), object(), _ThreadData(), temp_storage=object())
        with pytest.raises(TypeError, match="fixed-size ThreadData"):
            api.load(this_block(), object(), object())
        with pytest.raises(ValueError, match="must match the payload item count"):
            api.load(this_block(), object(), _ThreadData(length=1))
        with pytest.raises(TypeError, match="common API"):
            api.load(this_block(), object(), _ThreadData(dtype=np.float16))
        with pytest.raises(TypeError, match="common API"):
            api.store(this_block(), object(), np.complex64(1))
        warp_output = _ThreadData()
        assert api.load(this_warp(), object(), warp_output) is None
        api.store(this_warp(), object(), _ReadonlyThreadData())
        logical_output = _ThreadData()
        logical_warp = this_warp().group_by(8)
        assert api.load(logical_warp, object(), logical_output) is None
        api.store(logical_warp, object(), _ReadonlyThreadData())
        for group in (this_warp(), logical_warp):
            with pytest.raises(ValueError, match="supported only for block groups"):
                api.load(group, object(), _ThreadData(), algorithm="warp_transpose")
            with pytest.raises(ValueError, match="not supported for Warp groups"):
                api.store(
                    group, object(), _ReadonlyThreadData(), temp_storage=_TempStorage()
                )


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
def test_common_load_store_accept_every_advertised_dtype(monkeypatch, dtype):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    calls = []
    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda operation, *args, **kwargs: calls.append(operation),
    )
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.load_store"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        output = _ThreadData(dtype=dtype)
        assert api.load(this_block(), object(), output) is None
        api.store(this_block(), object(), dtype(1))
    assert calls == ["load", "store"]


def test_common_static_controls_fail_closed_before_delegation(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    calls = []
    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.load_store"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        for kwargs, exception, message in [
            (
                {"valid_items": 1.5},
                TypeError,
                "integer value supported by the common API",
            ),
            ({"offset": "4"}, TypeError, "integer value supported by the common API"),
            ({"valid_items": -1}, ValueError, "between 0"),
            ({"offset": -1}, ValueError, "between 0"),
            ({"offset": 1 << 63}, ValueError, "between 0"),
        ]:
            with pytest.raises(exception, match=message):
                api.load(this_block(), object(), _ThreadData(), **kwargs)
    assert calls == []

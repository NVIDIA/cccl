# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common scan frontend checks for CUTLASS tracing."""

from enum import Enum
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from cuda.coop._core import this_block, this_warp

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _ThreadData:
    def __init__(self, items_per_thread=2, *, dtype=np.float32, length=None):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [np.float32(0)] * (
            items_per_thread if length is None else length
        )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _TempStorage:
    size_in_bytes = 128
    alignment = 16
    auto_sync = True
    sharing = "shared"


class _StringSelector(str, Enum):
    INCLUSIVE = "inclusive"
    RAKING = "raking"
    MAX = "max"


def test_common_scan_validates_payload_and_option_matrix(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.scan")
    delegated = object()
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return delegated

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.scan"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        assert api.inclusive_sum(this_block(), _ThreadData()) is delegated
        assert api.scan(this_warp(), np.float32(1)) is delegated
        assert (
            api.exclusive_scan(
                this_block(),
                np.float32(1),
                scan_op=" maximum ",
                initial_value=0,
                algorithm="raking-memoize",
                temp_storage=_TempStorage(),
            )
            is delegated
        )
        with pytest.raises(
            TypeError, match="numeric scalar for warp scans in the common API"
        ):
            api.inclusive_sum(this_warp(), _ThreadData())
        with pytest.raises(ValueError, match="require initial_value"):
            api.exclusive_scan(this_block(), np.int32(1), scan_op="max")
        with pytest.raises(ValueError, match="only for blocks"):
            api.inclusive_sum(this_warp(), np.int32(1), algorithm="raking")
        with pytest.raises(ValueError, match="only for blocks"):
            api.inclusive_sum(this_warp(), np.int32(1), temp_storage=object())
        with pytest.raises(TypeError, match="must satisfy TempStorageLike"):
            api.inclusive_sum(this_block(), np.int32(1), temp_storage=object())
        with pytest.raises(TypeError, match="value dtypes"):
            api.inclusive_scan(this_block(), np.float32(1), scan_op="bit_and")
    assert len(calls) == 3
    assert calls[2][1]["scan_op"] == "max"
    assert calls[2][1]["algorithm"] == "raking_memoize"


def test_common_scan_rejects_inclusive_initial_before_delegation(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.scan")
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
            import_module("cuda.coop._core.api.scan"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises(ValueError, match="not supported for inclusive"):
            api.scan(
                this_block(), np.int32(1), mode="inclusive", initial_value=0
            )
    assert calls == []


@pytest.mark.parametrize(
    ("parameter", "token"),
    (("mode", "inclusive"), ("algorithm", "raking"), ("scan_op", "max")),
)
@pytest.mark.parametrize("selector_kind", ("object", "string-enum"))
def test_common_python_entry_point_rejects_non_string_scan_selectors(
    monkeypatch, parameter, token, selector_kind
):
    import importlib

    from cuda.coop._core.api import _dispatch
    from cuda.coop._core.api.thread_group import this_block

    scan_api = importlib.import_module("cuda.coop._core.api.scan")
    group = this_block()
    selector = (
        SimpleNamespace(value=token)
        if selector_kind == "object"
        else _StringSelector(token)
    )
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            _dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.scan"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises(TypeError, match=f"{parameter} must be .*string"):
            scan_api.scan(group, np.int32(1), **{parameter: selector})

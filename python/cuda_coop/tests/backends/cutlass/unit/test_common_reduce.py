# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common reduce frontend checks for CUTLASS tracing."""

from enum import Enum
from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core import (
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _StringSelector(str, Enum):
    RAKING = "raking"
    MAXIMUM = "maximum"


class _ThreadData:
    items_per_thread = 2
    dtype = np.float32

    def __init__(self):
        self._items = [np.float32(1), np.float32(2)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def test_common_reduce_matrix_and_family_owned_selectors(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.reduce")
    delegated = object()
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return delegated

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    groups = (
        this_thread(),
        this_warp(),
        this_warp().group_by(8),
        this_block(),
        this_block().group_by(2),
        this_cluster(),
    )
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.reduce"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        for group in groups:
            assert api.reduce(group, _ThreadData(), binary_op="+") is delegated
            assert api.sum(group, np.int32(1), broadcast=False) is delegated
        assert calls[0][1]["binary_op"] == "sum"
        assert (
            api.reduce(this_block(), np.int32(1), binary_op=" MAXIMUM ")
            is delegated
        )
        assert calls[-1][1]["binary_op"] == "max"
        assert (
            api.sum(
                this_block(),
                np.int32(1),
                broadcast=False,
                algorithm=" RAKING-COMMUTATIVE-ONLY ",
            )
            is delegated
        )
        assert calls[-1][1]["algorithm"] == "raking_commutative_only"
        with pytest.raises(
            NotImplementedError, match="hidden per-launch workspace"
        ):
            api.sum(this_grid(), np.int32(1))
        with pytest.raises(TypeError, match="value dtypes"):
            api.reduce(this_block(), np.float32(1), binary_op="bit_and")
        with pytest.raises(TypeError, match="binary_op must be a string"):
            api.reduce(this_block(), np.int32(1), binary_op=object())


@pytest.mark.parametrize("selector", [0, _StringSelector.RAKING])
def test_common_reduce_algorithm_rejects_non_string_selectors(
    monkeypatch, selector
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.reduce")
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
            import_module("cuda.coop._core.api.reduce"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises(TypeError, match="algorithm must be a string"):
            api.sum(
                this_block(), np.int32(1), broadcast=False, algorithm=selector
            )
    assert calls == []


@pytest.mark.parametrize(
    "selector", [0, _StringSelector.MAXIMUM, lambda x, y: x]
)
def test_common_reduce_operator_rejects_non_string_selectors(
    monkeypatch, selector
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.reduce")
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
            import_module("cuda.coop._core.api.reduce"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises(TypeError, match="binary_op must be a string"):
            api.reduce(this_block(), np.int32(1), binary_op=selector)
    assert calls == []


@pytest.mark.parametrize("operation", ["reduce", "sum"])
def test_common_cub_controls_fail_closed_before_delegation(
    monkeypatch, operation
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.reduce")
    function = getattr(api, operation)
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
            import_module("cuda.coop._core.api.reduce"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises(ValueError, match="requires broadcast=False"):
            function(this_block(), np.int32(1), valid_items=17)
        with pytest.raises(ValueError, match="scalar values only"):
            function(
                this_block(), _ThreadData(), broadcast=False, valid_items=1
            )
        with pytest.raises(ValueError, match="requires a block group"):
            function(
                this_warp(), np.int32(1), broadcast=False, algorithm="raking"
            )
        with pytest.raises(ValueError, match="at least 1"):
            function(this_block(), np.int32(1), broadcast=False, valid_items=0)
    assert calls == []

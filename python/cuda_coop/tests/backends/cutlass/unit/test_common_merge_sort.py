# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common merge sort frontend checks for CUTLASS tracing."""

from importlib import import_module

import numpy as np
import pytest

from cuda import coop
from cuda.coop._core import this_block, this_warp

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _ReadonlyThreadData:
    def __init__(self, items, dtype):
        self._items = list(items)
        self.items_per_thread = len(self._items)
        self.dtype = dtype

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _ThreadData(_ReadonlyThreadData):
    def __setitem__(self, index, value):
        self._items[index] = value


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(this_block(), id="block"),
        pytest.param(this_warp(), id="physical-warp"),
        pytest.param(this_warp().group_by(8), id="logical-warp"),
    ],
)
@pytest.mark.parametrize(
    "operation,payload_types",
    [
        pytest.param("merge_sort_keys", (_ReadonlyThreadData,), id="keys"),
        pytest.param(
            "merge_sort_pairs", (_ReadonlyThreadData, _ThreadData), id="pair-keys"
        ),
        pytest.param(
            "merge_sort_pairs", (_ThreadData, _ReadonlyThreadData), id="pair-values"
        ),
        pytest.param(
            "merge_sort_pairs",
            (_ReadonlyThreadData, _ReadonlyThreadData),
            id="pair-both",
        ),
    ],
)
def test_common_merge_sort_accepts_readonly_inputs(
    monkeypatch, group, operation, payload_types
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.merge_sort")
    payloads = tuple(
        (
            payload_type(items, dtype)
            for payload_type, items, dtype in zip(
                payload_types, ((3, 1, 2), (0.3, 0.1, 0.2)), (np.int32, np.float64)
            )
        )
    )
    original_items = [list(payload) for payload in payloads]
    calls = []
    result = object()

    def marker(selected_operation, selected_group, *selected_payloads, **kwargs):
        calls.append((selected_operation, selected_group, selected_payloads))
        return result

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.merge_sort"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        assert getattr(coop, operation)(group, *payloads) is result
    assert calls == [(operation, group, payloads)]
    assert [list(payload) for payload in payloads] == original_items

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral Merge Sort payload, topology, and validation contracts."""

from importlib import import_module

import numpy as np
import pytest

from cuda import coop
from cuda.coop._core import (
    FLOAT64,
    INT32,
    ArgumentBinding,
    CxxOperator,
    Dependency,
    GroupLoweringTarget,
    GroupMergeSortSemantics,
    LaunchFacts,
    StorageOwnership,
    make_block_merge_sort_semantics,
    make_block_merge_sort_spec,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_grid,
    this_warp,
)

_COMPARE = CxxOperator("::cuda::std::less<KeyT>", Dependency("KeyT"), "compare_op")


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
def test_portable_merge_sort_accepts_readonly_inputs(
    monkeypatch, group, operation, payload_types
):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.merge_sort")
    payloads = tuple(
        payload_type(items, dtype)
        for payload_type, items, dtype in zip(
            payload_types, ((3, 1, 2), (0.3, 0.1, 0.2)), (np.int32, np.float64)
        )
    )
    original_items = [list(payload) for payload in payloads]
    calls = []
    result = object()

    def marker(selected_operation, selected_group, *selected_payloads, **kwargs):
        calls.append((selected_operation, selected_group, selected_payloads))
        return result

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with dispatch._compiler_scope("test.backend"):
        assert getattr(coop, operation)(group, *payloads) is result

    assert calls == [(operation, group, payloads)]
    assert [list(payload) for payload in payloads] == original_items


def _operation(*, pairs=False, valid=None):
    return GroupMergeSortSemantics(
        make_block_merge_sort_semantics(
            key_dtype=INT32,
            value_dtype=FLOAT64 if pairs else None,
            items_per_thread=3,
            compare_operator=_COMPARE,
            valid_items=0 if valid is not None else None,
            oob_default=0 if valid is not None else None,
        ),
        valid_items=ArgumentBinding.omitted() if valid is None else valid,
    )


def _plan(group, **kwargs):
    return plan_group_primitive(
        make_group_primitive_call(group, _operation(**kwargs)), LaunchFacts((8, 4, 2))
    )


@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize(
    "group,width,target",
    [
        (this_block(), 64, GroupLoweringTarget.CUB_BLOCK),
        (this_warp(), 32, GroupLoweringTarget.CUB_WARP),
        (this_warp().group_by(8), 8, GroupLoweringTarget.CUB_WARP),
    ],
)
def test_results_and_group_scratch(pairs, group, width, target):
    plan = _plan(group, pairs=pairs).require_supported()
    assert plan.target is target
    assert [value.name for value in plan.result.values] == (
        ["keys", "values"] if pairs else ["keys"]
    )
    assert [value.dtype for value in plan.result.values] == (
        [INT32, FLOAT64] if pairs else [INT32]
    )
    assert all(value.items_per_member == 3 for value in plan.result.values)
    assert plan.topology.instances == 64 // width
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION


@pytest.mark.parametrize("count", [-1, 193, 1 << 32])
def test_invalid_static_count_is_rejected(count):
    with pytest.raises(ValueError, match="valid_items"):
        _plan(this_block(), valid=ArgumentBinding.static(count))


def test_runtime_count_has_range_contract_and_guard():
    plan = _plan(
        this_warp().group_by(8), valid=ArgumentBinding.runtime()
    ).require_supported()
    requirement = plan.participation.argument_preconditions[0]
    assert (requirement.minimum, requirement.maximum) == (0, 24)
    source = "\n".join(
        definition.code for definition in plan.implementation.type_definitions
    )
    assert source.index("valid_items > TileSize") < source.index(
        "static_cast<int>(valid_items)"
    )


def test_partial_count_participates_in_semantic_identity():
    assert _operation(valid=ArgumentBinding.static(5)) != _operation(
        valid=ArgumentBinding.static(6)
    )
    assert _operation(valid=ArgumentBinding.runtime()) == _operation(
        valid=ArgumentBinding.runtime()
    )


@pytest.mark.parametrize("group", [this_grid(), this_block().group_by(2)])
def test_unsupported_group(group):
    assert _plan(group).unsupported is not None


def test_non_power_of_two_block_is_unsupported():
    plan = plan_group_primitive(
        make_group_primitive_call(this_block(), _operation()), LaunchFacts(48)
    )
    with pytest.raises(Exception, match="power-of-two"):
        plan.require_supported()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"items_per_thread": 0},
        {"items_per_thread": True},
        {"valid_items": 1},
        {"oob_default": 99},
        {"compare_operator": None},
    ],
)
def test_primitive_controls(kwargs):
    controls = dict(key_dtype=INT32, items_per_thread=3, compare_operator=_COMPARE)
    controls.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        make_block_merge_sort_semantics(**controls)


@pytest.mark.parametrize("block_dim", [(8, True, 1), (8, 4.0, 1), (0, 1, 1)])
def test_invalid_block_dimensions(block_dim):
    with pytest.raises(ValueError, match="three positive dimensions"):
        make_block_merge_sort_spec(
            key_dtype=INT32,
            items_per_thread=3,
            compare_operator=_COMPARE,
            block_dim=block_dim,
        )

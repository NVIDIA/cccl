# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable Block Load/Store planning and root API contracts."""

from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    GroupLoadStoreAlgorithm,
    GroupLoweringTarget,
    LaunchFactOrigin,
    LaunchFacts,
    ParameterRole,
    PreconditionEnforcement,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    make_group_primitive_call,
    plan_group_primitive,
    resolve_thread_group,
    this_block,
    this_grid,
    this_thread,
    this_warp,
)
from cuda.coop._core.group._contracts import _cub_warp_width, _group_topology
from tests.support.group_planning import _load_store, _plan


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


def test_portable_load_returns_the_identical_output_alias(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    output = _ThreadData()
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return args[3]

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with dispatch._compiler_scope("test.backend"):
        assert api.load(this_block(), object(), output) is output

    assert len(calls) == 1


def test_portable_load_store_validate_block_payloads_and_options(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda operation, *args, **kwargs: args[2] if operation == "load" else None,
    )

    with dispatch._compiler_scope("test.backend"):
        api.store(
            this_block(),
            object(),
            _ReadonlyThreadData(),
            temp_storage=_TempStorage(),
        )
        with pytest.raises(ValueError, match="oob_default requires valid_items"):
            api.load(this_block(), object(), _ThreadData(), oob_default=0)
        with pytest.raises(TypeError, match="must satisfy TempStorageLike"):
            api.load(
                this_block(),
                object(),
                _ThreadData(),
                temp_storage=object(),
            )
        with pytest.raises(TypeError, match="fixed-size ThreadData"):
            api.load(this_block(), object(), object())
        with pytest.raises(ValueError, match="must match the payload item count"):
            api.load(this_block(), object(), _ThreadData(length=1))
        with pytest.raises(TypeError, match="portable API"):
            api.load(this_block(), object(), _ThreadData(dtype=np.float16))
        with pytest.raises(TypeError, match="portable API"):
            api.store(this_block(), object(), np.complex64(1))
        with pytest.raises(NotImplementedError, match="group kind 'physical_warp'"):
            api.load(this_warp(), object(), _ThreadData())


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
def test_portable_load_store_accept_every_advertised_dtype(monkeypatch, dtype):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    calls = []
    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda operation, *args, **kwargs: (
            calls.append(operation) or (args[2] if operation == "load" else None)
        ),
    )

    with dispatch._compiler_scope("test.backend"):
        output = _ThreadData(dtype=dtype)
        assert api.load(this_block(), object(), output) is output
        api.store(this_block(), object(), dtype(1))

    assert calls == ["load", "store"]


def test_portable_static_controls_fail_closed_before_delegation(monkeypatch):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    calls = []
    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with dispatch._compiler_scope("test.backend"):
        for kwargs, exception, message in [
            ({"valid_items": 1.5}, TypeError, "portable integer"),
            ({"offset": "4"}, TypeError, "portable integer"),
            ({"valid_items": -1}, ValueError, "between 0"),
            ({"offset": -1}, ValueError, "between 0"),
            ({"offset": 1 << 63}, ValueError, "between 0"),
        ]:
            with pytest.raises(exception, match=message):
                api.load(this_block(), object(), _ThreadData(), **kwargs)

    assert calls == []


def test_group_load_and_store_select_complete_block_contracts():
    load = _plan(this_block(), _load_store("load", items_per_thread=3), (8, 4, 1))
    store = _plan(this_block(), _load_store("store", items_per_thread=3), (8, 4, 1))

    assert load.target is GroupLoweringTarget.CUB_BLOCK
    assert load.provenance.cpp_class == "cub::BlockLoad"
    assert load.result.result_items_per_thread == 3
    assert load.resolved_group.hierarchy.block_dim == (8, 4, 1)
    assert store.target is GroupLoweringTarget.CUB_BLOCK
    assert store.provenance.cpp_class == "cub::BlockStore"
    assert store.result is None
    assert load.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert load.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert load.topology.group_kind == "block"
    assert load.topology.logical_width == 32
    assert load.topology.instances == 1
    assert load.topology.instance_index == "cta"
    assert load.topology.execution_scope is SynchronizationScope.BLOCK


@pytest.mark.parametrize(
    ("group", "logical_width", "instances", "index", "scope"),
    [
        (this_thread(), 1, 64, "linear_thread_rank", SynchronizationScope.NONE),
        (this_warp(), 32, 2, "linear_thread_rank / 32", SynchronizationScope.WARP),
        (
            this_warp().group_by(8),
            8,
            8,
            "linear_thread_rank / 8",
            SynchronizationScope.WARP,
        ),
        (
            this_block().group_by(1),
            32,
            2,
            "linear_thread_rank / 32",
            SynchronizationScope.WARP,
        ),
        (
            this_block().group_by(2),
            64,
            1,
            "linear_thread_rank / 64",
            SynchronizationScope.GROUP,
        ),
        (this_block(), 64, 1, "cta", SynchronizationScope.BLOCK),
    ],
)
def test_group_topology_is_family_independent(
    group,
    logical_width,
    instances,
    index,
    scope,
):
    launch = LaunchFacts(exact_block_dim=64)
    resolved = resolve_thread_group(group, launch).require_supported()

    topology = _group_topology(resolved, launch)

    assert topology.group_kind == resolved.kind
    assert topology.logical_width == logical_width
    assert topology.instances == instances
    assert topology.instance_index == index
    assert topology.execution_scope is scope


def test_shared_cub_warp_width_rules():
    assert _cub_warp_width(this_warp()) == 32
    assert _cub_warp_width(this_warp().group_by(8)) == 8
    with pytest.raises(ValueError, match="power-of-two"):
        _cub_warp_width(this_warp().group_by(3, exhaustive=False))
    with pytest.raises(ValueError, match="warp-based"):
        _cub_warp_width(this_block())


def test_grid_family_requires_verified_cooperative_launch():
    from dataclasses import dataclass

    from cuda.coop._core.group._dispatch import _register_group_operation_family

    @dataclass(frozen=True)
    class _GridOperation:
        result_visibility = ResultVisibility.ALL_MEMBERS
        returns_value = False

        @property
        def semantic_key(self):
            return ("test-grid-operation",)

    marker = object()
    _register_group_operation_family(
        _GridOperation,
        classifications=lambda _operation: (),
        planner=lambda *_args: marker,
        group_kinds=frozenset({"grid"}),
        unsupported_group_message="test operation requires a grid",
    )
    call = make_group_primitive_call(this_grid(), _GridOperation())
    asserted = LaunchFacts(
        exact_block_dim=64,
        exact_grid_dim=2,
        exact_cluster_dim=1,
        cluster_launch=False,
        cooperative_launch=True,
        provenance=LaunchFactOrigin(
            "cluster_launch",
            "test",
            verified=True,
        ),
    )

    unsupported = plan_group_primitive(call, asserted)

    assert unsupported.target is GroupLoweringTarget.UNSUPPORTED
    assert unsupported.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert "verified cooperative launch" in unsupported.unsupported.message

    verified = LaunchFacts(
        exact_block_dim=64,
        exact_grid_dim=2,
        exact_cluster_dim=1,
        cluster_launch=False,
        cooperative_launch=True,
        provenance=(
            LaunchFactOrigin("cluster_launch", "test", verified=True),
            LaunchFactOrigin("cooperative_launch", "test", verified=True),
        ),
    )
    assert plan_group_primitive(call, verified) is marker


def test_group_load_models_tile_controls_and_caller_storage():
    operation = _load_store(
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(0),
        offset=ArgumentBinding.static(4),
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="shared",
        storage_size_in_bytes=256,
        storage_alignment=16,
        storage_auto_sync=True,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.participation.valid_member_selection == (
        "first valid_items tile elements"
    )
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
        "offset",
    )
    assert [
        (item.name, item.kind, item.role) for item in call.argument_classifications
    ] == [
        ("source", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        ("valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("oob_default", ArgumentKind.STATIC, ParameterRole.CONSTANT),
        ("offset", ArgumentKind.STATIC, ParameterRole.CONSTANT),
        ("algorithm", ArgumentKind.STATIC, ParameterRole.CONSTANT),
    ]
    valid_items_precondition, offset_precondition = (
        plan.participation.argument_preconditions
    )
    assert (valid_items_precondition.minimum, valid_items_precondition.maximum) == (
        0,
        128,
    )
    assert valid_items_precondition.enforcement is PreconditionEnforcement.CALLER
    assert (offset_precondition.minimum, offset_precondition.maximum) == (
        0,
        (1 << 63) - 1,
    )
    assert offset_precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED
    assert plan.temp_storage.ownership is StorageOwnership.CALLER
    assert plan.temp_storage.sharing == "shared"
    assert plan.temp_storage.requested_size_in_bytes == 256
    assert plan.temp_storage.requested_alignment == 16
    assert plan.temp_storage.auto_sync


def test_storage_contract_is_part_of_semantic_and_artifact_identity():
    shared = _load_store(
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="shared",
        storage_auto_sync=True,
    )
    exclusive = _load_store(
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="exclusive",
        storage_auto_sync=False,
    )

    shared_plan = _plan(this_block(), shared)
    exclusive_plan = _plan(this_block(), exclusive)

    assert shared.semantic_key != exclusive.semantic_key
    assert shared_plan.semantic_key != exclusive_plan.semantic_key
    assert shared_plan.artifact_key != exclusive_plan.artifact_key
    assert (
        exclusive_plan.synchronization.storage_reuse_barrier
        is SynchronizationScope.NONE
    )


def test_valid_items_counts_the_entire_block_tile_and_accepts_zero():
    for value in (0, 128):
        plan = _plan(
            this_block(),
            _load_store(valid_items=ArgumentBinding.static(value)),
            64,
        )
        condition = plan.participation.argument_preconditions[0]
        assert condition.maximum == 128
        assert condition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED

    for value in (-1, 129):
        with pytest.raises(ValueError, match="group tile size"):
            _plan(
                this_block(),
                _load_store(valid_items=ArgumentBinding.static(value)),
                64,
            )


def test_static_control_identity_normalizes_numpy_integers():
    plain = _load_store(
        valid_items=ArgumentBinding.static(5),
        offset=ArgumentBinding.static(7),
    )
    numpy = _load_store(
        valid_items=ArgumentBinding.static(np.int32(5)),
        offset=ArgumentBinding.static(np.int64(7)),
    )

    assert plain.semantic_key == numpy.semantic_key
    assert (
        _plan(this_block(), plain).artifact_key
        == _plan(this_block(), numpy).artifact_key
    )


def test_group_load_store_rejects_negative_static_offsets():
    with pytest.raises(ValueError, match="static offset must be nonnegative"):
        _load_store(offset=ArgumentBinding.static(-1))

    runtime = _plan(
        this_block(),
        _load_store(offset=ArgumentBinding.runtime()),
    )
    offset_precondition = runtime.participation.argument_preconditions[0]
    assert offset_precondition.minimum == 0
    assert offset_precondition.enforcement is PreconditionEnforcement.CALLER


def test_complete_algorithm_enum_is_preserved_but_non_block_targets_are_unsupported():
    assert {algorithm.value for algorithm in GroupLoadStoreAlgorithm} == {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }

    unsupported = _plan(this_thread(), _load_store())
    assert unsupported.target is GroupLoweringTarget.UNSUPPORTED
    assert unsupported.unsupported.code is UnsupportedReasonCode.GROUP_KIND

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable Block Load/Store planning and root API contracts."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core import (
    INT64,
    ArgumentBinding,
    ArgumentKind,
    GroupLoadStoreAlgorithm,
    GroupLoweringTarget,
    GroupTopologyContract,
    LaunchFactOrigin,
    LaunchFacts,
    ParameterRole,
    PointerOffset,
    PreconditionEnforcement,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    TempStorageContract,
    UnsupportedReasonCode,
    make_group_primitive_call,
    plan_group_primitive,
    resolve_thread_group,
    this_block,
    this_grid,
    this_thread,
    this_warp,
)
from cuda.coop._core.group._contracts import (
    _contracts,
    _cub_warp_width,
    _group_topology,
)
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


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(this_block(), id="block"),
        pytest.param(this_warp(), id="physical-warp"),
        pytest.param(this_warp().group_by(8), id="logical-warp"),
    ],
)
def test_portable_load_returns_the_identical_output_alias(monkeypatch, group):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.load_store")
    output = _ThreadData()
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return args[3]

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with dispatch._compiler_scope("test.backend"):
        assert api.load(group, object(), output) is output

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
        warp_output = _ThreadData()
        assert api.load(this_warp(), object(), warp_output) is warp_output
        api.store(this_warp(), object(), _ReadonlyThreadData())
        logical_output = _ThreadData()
        logical_warp = this_warp().group_by(8)
        assert api.load(logical_warp, object(), logical_output) is logical_output
        api.store(logical_warp, object(), _ReadonlyThreadData())
        for group in (this_warp(), logical_warp):
            with pytest.raises(ValueError, match="supported only for block groups"):
                api.load(
                    group,
                    object(),
                    _ThreadData(),
                    algorithm="warp_transpose",
                )
            with pytest.raises(ValueError, match="not supported for Warp groups"):
                api.store(
                    group,
                    object(),
                    _ReadonlyThreadData(),
                    temp_storage=_TempStorage(),
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
    assert load.temp_storage.ownership is StorageOwnership.NONE
    assert load.temp_storage.address_space is None
    assert load.temp_storage.cpp_type is None
    assert load.temp_storage.instances is None
    assert load.temp_storage.instance_index is None
    assert not load.temp_storage.exact_layout_required
    assert load.temp_storage.sharing is None
    assert load.temp_storage.requested_size_in_bytes is None
    assert load.temp_storage.requested_alignment is None
    assert not load.temp_storage.auto_sync
    assert load.synchronization.storage_reuse_barrier is SynchronizationScope.NONE
    assert load.topology.group_kind == "block"
    assert load.topology.logical_width == 32
    assert load.topology.instances == 1
    assert load.topology.instance_index == "cta"
    assert load.topology.thread_rank == "linear_thread_rank"
    assert load.topology.execution_scope is SynchronizationScope.BLOCK


def test_partial_transpose_load_records_preserving_wrapper_provenance():
    plan = _plan(
        this_block(),
        _load_store(
            "load",
            algorithm=GroupLoadStoreAlgorithm.TRANSPOSE,
            valid_items=ArgumentBinding.runtime(),
        ),
    )

    assert plan.provenance.cpp_class == ("cub::CudaCoopBlockLoadPreservingInvalid")


def test_physical_warp_load_store_select_complete_cub_contracts():
    load = _plan(
        this_warp(),
        _load_store("load", items_per_thread=3),
        (64, 1, 1),
    )
    store = _plan(
        this_warp(),
        _load_store("store", items_per_thread=3),
        (64, 1, 1),
    )

    assert load.target is GroupLoweringTarget.CUB_WARP
    assert load.provenance.header == "cub/warp/warp_load.cuh"
    assert load.provenance.cpp_class == "cub::WarpLoad"
    assert load.result.result_items_per_thread == 3
    assert store.target is GroupLoweringTarget.CUB_WARP
    assert store.provenance.cpp_class == "cub::WarpStore"
    assert store.result is None
    assert load.topology.group_kind == "warp"
    assert load.topology.logical_width == 32
    assert load.topology.instances == 2
    assert load.topology.instance_index == "linear_thread_rank / 32"
    assert load.topology.thread_rank == "linear_thread_rank % 32"
    assert load.topology.execution_scope is SynchronizationScope.WARP
    assert load.temp_storage.ownership is StorageOwnership.NONE
    assert load.synchronization.storage_reuse_barrier is SynchronizationScope.NONE


@pytest.mark.parametrize("logical_width", [1, 2, 4, 8, 16, 32])
def test_logical_warp_load_selects_width_specific_cub_contract(logical_width):
    plan = _plan(
        this_warp().group_by(logical_width),
        _load_store("load", items_per_thread=3),
        64,
    )

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.provenance.header == "cub/warp/warp_load.cuh"
    assert plan.provenance.cpp_class == "cub::WarpLoad"
    assert plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == (
        logical_width
    )
    assert plan.result.result_items_per_thread == 3
    assert plan.topology.group_kind == "threads_within_warp"
    assert plan.topology.logical_width == logical_width
    assert plan.topology.instances == 64 // logical_width
    assert plan.topology.instance_index == (f"linear_thread_rank / {logical_width}")
    assert plan.topology.thread_rank == f"linear_thread_rank % {logical_width}"
    assert plan.topology.execution_scope is SynchronizationScope.WARP
    assert plan.temp_storage.ownership is StorageOwnership.NONE
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE


@pytest.mark.parametrize("kind", ("load", "store"))
@pytest.mark.parametrize(
    ("group", "instances", "instance_index"),
    [
        pytest.param(
            this_warp(),
            2,
            "linear_thread_rank / 32",
            id="physical-warp",
        ),
        pytest.param(
            this_warp().group_by(8),
            8,
            "linear_thread_rank / 8",
            id="logical-warp",
        ),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    (
        GroupLoadStoreAlgorithm.DIRECT,
        GroupLoadStoreAlgorithm.STRIPED,
        GroupLoadStoreAlgorithm.VECTORIZE,
        GroupLoadStoreAlgorithm.TRANSPOSE,
    ),
)
def test_warp_algorithm_storage_contract_matches_cub(
    kind,
    group,
    instances,
    instance_index,
    algorithm,
):
    plan = _plan(group, _load_store(kind, algorithm=algorithm), 64)
    storage_free = algorithm is not GroupLoadStoreAlgorithm.TRANSPOSE

    assert plan.temp_storage.ownership is (
        StorageOwnership.NONE if storage_free else StorageOwnership.IMPLEMENTATION
    )
    assert plan.temp_storage.instances == (None if storage_free else instances)
    assert plan.temp_storage.instance_index == (
        None if storage_free else instance_index
    )
    assert plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.NONE if storage_free else SynchronizationScope.WARP
    )


@pytest.mark.parametrize(
    ("group", "logical_width"),
    [
        pytest.param(this_warp(), 32, id="physical-warp"),
        pytest.param(this_warp().group_by(8), 8, id="logical-warp"),
    ],
)
def test_partial_warp_transpose_load_records_preserving_wrapper(
    group,
    logical_width,
):
    plan = _plan(
        group,
        _load_store(
            "load",
            algorithm=GroupLoadStoreAlgorithm.TRANSPOSE,
            valid_items=ArgumentBinding.runtime(),
        ),
        64,
    )

    assert plan.provenance.cpp_class == ("cub::CudaCoopWarpLoadPreservingInvalid")
    assert plan.implementation.metadata["preserves_invalid_items"]
    assert (
        plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == logical_width
    )


def test_physical_warp_plan_preserves_user_offset_and_requires_effective_offset():
    operation = _load_store(offset=ArgumentBinding.static(7))
    plan = _plan(this_warp(), operation, 64)

    assert plan.call.operation.offset == ArgumentBinding.static(7)
    assert plan.implementation.metadata["requires_runtime_effective_offset"]
    assert plan.implementation.metadata["effective_offset_origin"] == ("group_instance")
    assert plan.implementation.metadata["effective_offset_stride"] == 64
    provider_offset = plan.implementation.parameters[0][-1]
    assert provider_offset == PointerOffset(
        INT64,
        name="offset",
        pointer_arg_index=0,
    )
    offset_precondition = plan.participation.argument_preconditions[0]
    assert (offset_precondition.minimum, offset_precondition.maximum) == (
        0,
        (1 << 63) - 1 - 64,
    )
    assert offset_precondition.enforcement is (
        PreconditionEnforcement.PLANNER_VALIDATED
    )

    with pytest.raises(ValueError, match="tile origin must fit"):
        _plan(
            this_warp(),
            _load_store(offset=ArgumentBinding.static((1 << 63) - 1)),
            64,
        )


def test_logical_warp_plan_accounts_for_every_group_in_effective_offset():
    operation = _load_store(offset=ArgumentBinding.static(7))
    plan = _plan(this_warp().group_by(8), operation, 64)

    assert plan.call.operation.offset == ArgumentBinding.static(7)
    assert plan.topology.instances == 8
    assert plan.implementation.metadata["requires_runtime_effective_offset"]
    assert plan.implementation.metadata["effective_offset_origin"] == ("group_instance")
    assert plan.implementation.metadata["effective_offset_stride"] == 16
    provider_offset = plan.implementation.parameters[0][-1]
    assert provider_offset == PointerOffset(
        INT64,
        name="offset",
        pointer_arg_index=0,
    )
    offset_precondition = plan.participation.argument_preconditions[0]
    assert (offset_precondition.minimum, offset_precondition.maximum) == (
        0,
        (1 << 63) - 1 - 112,
    )
    assert offset_precondition.enforcement is (
        PreconditionEnforcement.PLANNER_VALIDATED
    )

    with pytest.raises(ValueError, match="warp-group tile origin"):
        _plan(
            this_warp().group_by(8),
            _load_store(offset=ArgumentBinding.static((1 << 63) - 1)),
            64,
        )


def test_physical_warp_valid_items_is_per_warp_tile():
    for value in (0, 64):
        plan = _plan(
            this_warp(),
            _load_store(valid_items=ArgumentBinding.static(value)),
            64,
        )
        condition = plan.participation.argument_preconditions[0]
        assert (condition.minimum, condition.maximum) == (0, 64)
        assert condition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED

    with pytest.raises(ValueError, match=r"group tile size \(64\)"):
        _plan(
            this_warp(),
            _load_store(valid_items=ArgumentBinding.static(65)),
            64,
        )

    runtime = _plan(
        this_warp(),
        _load_store(valid_items=ArgumentBinding.runtime()),
        64,
    )
    condition = runtime.participation.argument_preconditions[0]
    assert (condition.minimum, condition.maximum) == (0, 64)
    assert condition.enforcement is PreconditionEnforcement.CALLER


def test_logical_warp_valid_items_is_per_logical_group_tile():
    logical_warp = this_warp().group_by(8)
    for value in (0, 16):
        plan = _plan(
            logical_warp,
            _load_store(valid_items=ArgumentBinding.static(value)),
            64,
        )
        condition = plan.participation.argument_preconditions[0]
        assert (condition.minimum, condition.maximum) == (0, 16)
        assert condition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED

    for value in (-1, 17):
        with pytest.raises(ValueError, match=r"group tile size \(16\)"):
            _plan(
                logical_warp,
                _load_store(valid_items=ArgumentBinding.static(value)),
                64,
            )

    runtime = _plan(
        logical_warp,
        _load_store(valid_items=ArgumentBinding.runtime()),
        64,
    )
    condition = runtime.participation.argument_preconditions[0]
    assert (condition.minimum, condition.maximum) == (0, 16)
    assert condition.enforcement is PreconditionEnforcement.CALLER


def test_physical_warp_transpose_carries_per_instance_caller_storage_contract():
    plan = _plan(
        this_warp(),
        _load_store(
            algorithm=GroupLoadStoreAlgorithm.TRANSPOSE,
            storage_ownership=StorageOwnership.CALLER,
            storage_sharing="shared",
            storage_size_in_bytes=256,
            storage_alignment=16,
            storage_auto_sync=True,
        ),
        64,
    )

    assert plan.temp_storage.ownership is StorageOwnership.CALLER
    assert plan.temp_storage.instances == 2
    assert plan.temp_storage.instance_index == "linear_thread_rank / 32"
    assert plan.temp_storage.exact_layout_required
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP


@pytest.mark.parametrize(
    "algorithm",
    (
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE,
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    ),
)
@pytest.mark.parametrize(
    "group",
    [
        pytest.param(this_warp(), id="physical-warp"),
        pytest.param(this_warp().group_by(8), id="logical-warp"),
    ],
)
def test_warp_groups_reject_block_only_algorithms(group, algorithm):
    plan = _plan(group, _load_store(algorithm=algorithm), 64)

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "does not support algorithm" in plan.unsupported.message


def test_load_store_rejects_invalid_logical_width_and_incomplete_warps():
    invalid_width = _plan(
        this_warp().group_by(3, exhaustive=False),
        _load_store(),
        64,
    )
    incomplete_physical = _plan(this_warp(), _load_store(), 48)
    incomplete_logical = _plan(this_warp().group_by(8), _load_store(), 48)

    assert invalid_width.target is GroupLoweringTarget.UNSUPPORTED
    assert invalid_width.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert "power-of-two group width" in invalid_width.unsupported.message
    for incomplete in (incomplete_physical, incomplete_logical):
        assert incomplete.target is GroupLoweringTarget.UNSUPPORTED
        assert (
            incomplete.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
        )


def test_every_block_algorithm_has_distinct_plan_and_artifact_identity():
    plans = [
        _plan(this_block(), _load_store("load", algorithm=algorithm))
        for algorithm in GroupLoadStoreAlgorithm
    ]

    assert len({plan.semantic_key for plan in plans}) == len(plans)
    assert len({plan.artifact_key for plan in plans}) == len(plans)


def test_warp_width_is_part_of_plan_and_artifact_identity():
    plans = [
        _plan(this_warp().group_by(8), _load_store(), 64),
        _plan(this_warp().group_by(16), _load_store(), 64),
        _plan(this_warp(), _load_store(), 64),
    ]

    assert len({plan.semantic_key for plan in plans}) == len(plans)
    assert len({plan.artifact_key for plan in plans}) == len(plans)
    assert [
        plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] for plan in plans
    ] == [8, 16, 32]


@pytest.mark.parametrize(
    ("group", "logical_width", "instances", "index", "rank", "scope"),
    [
        (
            this_thread(),
            1,
            64,
            "linear_thread_rank",
            "0",
            SynchronizationScope.NONE,
        ),
        (
            this_warp(),
            32,
            2,
            "linear_thread_rank / 32",
            "linear_thread_rank % 32",
            SynchronizationScope.WARP,
        ),
        (
            this_warp().group_by(8),
            8,
            8,
            "linear_thread_rank / 8",
            "linear_thread_rank % 8",
            SynchronizationScope.WARP,
        ),
        (
            this_block().group_by(1),
            32,
            2,
            "linear_thread_rank / 32",
            "linear_thread_rank % 32",
            SynchronizationScope.WARP,
        ),
        (
            this_block().group_by(2),
            64,
            1,
            "linear_thread_rank / 64",
            "linear_thread_rank % 64",
            SynchronizationScope.GROUP,
        ),
        (
            this_block(),
            64,
            1,
            "cta",
            "linear_thread_rank",
            SynchronizationScope.BLOCK,
        ),
    ],
)
def test_group_topology_is_family_independent(
    group,
    logical_width,
    instances,
    index,
    rank,
    scope,
):
    launch = LaunchFacts(exact_block_dim=64)
    resolved = resolve_thread_group(group, launch).require_supported()

    topology = _group_topology(resolved, launch)

    assert topology.group_kind == resolved.kind
    assert topology.logical_width == logical_width
    assert topology.instances == instances
    assert topology.instance_index == index
    assert topology.thread_rank == rank
    assert topology.execution_scope is scope


def test_group_topology_preserves_the_original_positional_contract():
    topology = GroupTopologyContract(
        "block",
        64,
        1,
        "cta",
        SynchronizationScope.BLOCK,
    )

    assert topology.thread_rank == "linear_thread_rank"
    assert topology.execution_scope is SynchronizationScope.BLOCK


@pytest.mark.parametrize(
    ("ownership", "auto_sync", "expected_auto_sync", "expected_barrier"),
    [
        (StorageOwnership.NONE, None, False, SynchronizationScope.NONE),
        (StorageOwnership.IMPLEMENTATION, None, True, SynchronizationScope.BLOCK),
        (StorageOwnership.IMPLEMENTATION, False, False, SynchronizationScope.NONE),
    ],
)
def test_group_contracts_derive_auto_sync_from_storage_ownership(
    ownership,
    auto_sync,
    expected_auto_sync,
    expected_barrier,
):
    launch = LaunchFacts(exact_block_dim=64)
    resolved = resolve_thread_group(this_block(), launch).require_supported()

    _, _, synchronization, storage = _contracts(
        resolved,
        launch,
        result=None,
        storage_ownership=ownership,
        cpp_type=None,
        auto_sync=auto_sync,
    )

    assert storage.auto_sync is expected_auto_sync
    assert synchronization.storage_reuse_barrier is expected_barrier


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


def test_group_load_models_tile_controls_without_storage():
    operation = _load_store(
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(0),
        offset=ArgumentBinding.static(4),
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
    assert plan.temp_storage.ownership is StorageOwnership.NONE
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE


@pytest.mark.parametrize("kind", ("load", "store"))
@pytest.mark.parametrize(
    "algorithm",
    (
        GroupLoadStoreAlgorithm.DIRECT,
        GroupLoadStoreAlgorithm.STRIPED,
        GroupLoadStoreAlgorithm.VECTORIZE,
    ),
)
def test_storage_free_descriptor_is_not_part_of_plan_identity(kind, algorithm):
    implicit = _load_store(kind, algorithm=algorithm)
    explicit = _load_store(
        kind,
        algorithm=algorithm,
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="shared",
        storage_size_in_bytes=256,
        storage_alignment=16,
        storage_auto_sync=True,
    )

    implicit_plan = _plan(this_block(), implicit)
    explicit_plan = _plan(this_block(), explicit)

    assert explicit.storage_ownership is StorageOwnership.CALLER
    assert explicit.storage_sharing == "shared"
    assert explicit.storage_size_in_bytes == 256
    assert explicit.storage_alignment == 16
    assert explicit.storage_auto_sync
    assert implicit == explicit
    assert implicit.semantic_key == explicit.semantic_key
    assert implicit_plan == explicit_plan
    assert implicit_plan.semantic_key == explicit_plan.semantic_key
    assert implicit_plan.artifact_key == explicit_plan.artifact_key


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"storage_sharing": "invalid"}, ValueError, "shared or exclusive"),
        ({"storage_size_in_bytes": 0}, ValueError, "positive integer"),
        ({"storage_alignment": True}, ValueError, "positive integer"),
        ({"storage_auto_sync": 1}, TypeError, "must be a bool"),
    ],
)
def test_direct_storage_metadata_is_validated_before_being_ignored(
    kwargs,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        _load_store(**kwargs)


@pytest.mark.parametrize(
    ("storage_kwargs", "expected_ownership"),
    [
        ({}, StorageOwnership.IMPLEMENTATION),
        (
            {
                "storage_ownership": StorageOwnership.CALLER,
                "storage_sharing": "shared",
                "storage_size_in_bytes": 256,
                "storage_alignment": 16,
                "storage_auto_sync": True,
            },
            StorageOwnership.CALLER,
        ),
    ],
)
def test_direct_semantics_can_be_replaced_with_a_storage_bearing_algorithm(
    storage_kwargs,
    expected_ownership,
):
    direct = _load_store(**storage_kwargs)

    transpose = replace(direct, algorithm=GroupLoadStoreAlgorithm.TRANSPOSE)
    plan = _plan(this_block(), transpose)

    assert transpose.storage_ownership is expected_ownership
    assert plan.temp_storage.ownership is expected_ownership
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


@pytest.mark.parametrize("kind", ("load", "store"))
@pytest.mark.parametrize("algorithm", tuple(GroupLoadStoreAlgorithm))
def test_algorithm_storage_contract_matches_cub(kind, algorithm):
    plan = _plan(this_block(), _load_store(kind, algorithm=algorithm))

    storage_free = algorithm in {
        GroupLoadStoreAlgorithm.DIRECT,
        GroupLoadStoreAlgorithm.STRIPED,
        GroupLoadStoreAlgorithm.VECTORIZE,
    }
    assert plan.temp_storage.ownership is (
        StorageOwnership.NONE if storage_free else StorageOwnership.IMPLEMENTATION
    )
    assert plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.NONE if storage_free else SynchronizationScope.BLOCK
    )


@pytest.mark.parametrize(
    ("instances", "instance_index", "message"),
    [
        pytest.param(None, "cta", "positive instance count", id="missing-count"),
        pytest.param(0, "cta", "positive instance count", id="zero-count"),
        pytest.param(True, "cta", "positive instance count", id="boolean-count"),
        pytest.param(1, None, "non-empty instance index", id="missing-index"),
        pytest.param(1, "", "non-empty instance index", id="empty-index"),
    ],
)
def test_storage_bearing_contract_requires_instance_layout(
    instances,
    instance_index,
    message,
):
    with pytest.raises(ValueError, match=message):
        TempStorageContract(
            ownership=StorageOwnership.IMPLEMENTATION,
            address_space="shared",
            cpp_type="TestStorage",
            instances=instances,
            instance_index=instance_index,
            exact_layout_required=False,
        )


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        pytest.param("topology-kind", "resolved group kind", id="topology-kind"),
        pytest.param(
            "participation-kind",
            "resolved group kind",
            id="participation-kind",
        ),
        pytest.param("topology-width", "resolved group size", id="topology-width"),
        pytest.param(
            "participation-width",
            "resolved group size",
            id="participation-width",
        ),
        pytest.param("block-dim", "resolved group", id="block-dim"),
    ],
)
def test_supported_plan_contracts_must_describe_resolved_group(contract, message):
    plan = _plan(this_block(), _load_store())
    changes = {
        "topology-kind": {
            "topology": replace(plan.topology, group_kind="warp"),
        },
        "participation-kind": {
            "participation": replace(plan.participation, group_kind="warp"),
        },
        "topology-width": {
            "topology": replace(plan.topology, logical_width=32),
        },
        "participation-width": {
            "participation": replace(plan.participation, exact_group_size=32),
        },
        "block-dim": {
            "participation": replace(
                plan.participation,
                exact_block_dim=(32, 1, 1),
            ),
        },
    }[contract]

    with pytest.raises(ValueError, match=message):
        replace(plan, **changes)


@pytest.mark.parametrize(
    "algorithm",
    (
        GroupLoadStoreAlgorithm.TRANSPOSE,
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE,
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    ),
)
def test_storage_bearing_contract_is_part_of_plan_identity(algorithm):
    shared = _load_store(
        algorithm=algorithm,
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="shared",
        storage_auto_sync=True,
    )
    exclusive = _load_store(
        algorithm=algorithm,
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="exclusive",
        storage_auto_sync=False,
    )

    shared_plan = _plan(this_block(), shared)
    exclusive_plan = _plan(this_block(), exclusive)

    assert shared.semantic_key != exclusive.semantic_key
    assert shared_plan.semantic_key != exclusive_plan.semantic_key
    assert shared_plan.artifact_key != exclusive_plan.artifact_key
    assert shared_plan.temp_storage.exact_layout_required
    assert (
        shared_plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    )
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

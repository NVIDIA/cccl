# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    BlockReduceAlgorithm,
    CudaxReturnKind,
    CxxOperator,
    Dependency,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupReduceSemantics,
    LaunchFactOrigin,
    LaunchFacts,
    ParameterRole,
    PreconditionEnforcement,
    PythonOperator,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    make_group_primitive_call,
    make_reduce_semantics,
    plan_group_primitive,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)


def _builtin_operator(name="plus"):
    cpp = {
        "plus": "::cuda::std::plus<T>",
        "multiplies": "::cuda::std::multiplies<T>",
        "min": "::cuda::minimum<T>",
        "max": "::cuda::maximum<T>",
        "bit_and": "::cuda::std::bit_and<T>",
        "bit_or": "::cuda::std::bit_or<T>",
        "bit_xor": "::cuda::std::bit_xor<T>",
    }[name]
    return CxxOperator(cpp, Dependency("T"), name="binary_op")


def _reduce(
    *,
    dtype="int32",
    operation="sum",
    value_kind="scalar",
    items_per_thread=1,
    reduce_operator=None,
    valid_items=ArgumentBinding.omitted(),
    broadcast=True,
    cub_algorithm=None,
):
    if operation == "reduce" and reduce_operator is None:
        reduce_operator = _builtin_operator()
    return GroupReduceSemantics(
        make_reduce_semantics(
            dtype=dtype,
            items_per_thread=items_per_thread,
            operation=operation,
            value_kind=value_kind,
            reduce_operator=reduce_operator,
            valid_items=valid_items,
        ),
        broadcast=broadcast,
        cub_algorithm=cub_algorithm,
    )


def _plan(group, operation, launch=64):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(make_group_primitive_call(group, operation), facts)


def _cluster_facts():
    return LaunchFacts(
        exact_block_dim=64,
        exact_cluster_dim=2,
        cluster_launch=True,
        provenance=(
            LaunchFactOrigin("exact_cluster_dim", "test", verified=True),
            LaunchFactOrigin("cluster_launch", "test", verified=True),
        ),
    )


@pytest.mark.parametrize(
    ("group", "facts", "width", "instances", "execution_scope"),
    [
        (this_thread(), LaunchFacts(64), 1, 64, SynchronizationScope.NONE),
        (this_warp(), LaunchFacts(64), 32, 2, SynchronizationScope.WARP),
        (
            this_warp().group_by(8),
            LaunchFacts(64),
            8,
            8,
            SynchronizationScope.WARP,
        ),
        (this_block(), LaunchFacts(64), 64, 1, SynchronizationScope.BLOCK),
        (
            this_block().group_by(2),
            LaunchFacts(128),
            64,
            2,
            SynchronizationScope.GROUP,
        ),
        (
            this_cluster(),
            _cluster_facts(),
            128,
            1,
            SynchronizationScope.GROUP,
        ),
    ],
)
@pytest.mark.parametrize("broadcast", [True, False])
def test_builtin_full_reduce_uses_storage_free_cudax_across_hierarchy(
    group,
    facts,
    width,
    instances,
    execution_scope,
    broadcast,
):
    operation = _reduce(
        operation="reduce",
        value_kind="array",
        items_per_thread=4,
        broadcast=broadcast,
    )
    plan = _plan(group, operation, facts)

    assert plan.target is GroupLoweringTarget.CUDAX_GROUP
    assert plan.implementation.overload == ("broadcasted" if broadcast else "root_only")
    assert plan.implementation.return_kind is (
        CudaxReturnKind.VALUE if broadcast else CudaxReturnKind.OPTIONAL_VALUE
    )
    assert [parameter.name for parameter in plan.implementation.parameters] == [
        "item0",
        "item1",
        "item2",
        "item3",
    ]
    assert plan.topology.logical_width == width
    assert plan.topology.instances == instances
    assert plan.topology.execution_scope is execution_scope
    assert plan.temp_storage.ownership is StorageOwnership.NONE
    assert plan.temp_storage.address_space is None
    assert plan.temp_storage.instances is None
    assert not plan.temp_storage.auto_sync
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE
    assert plan.provenance.library == "CUDAX"


@pytest.mark.parametrize(
    "operator_name",
    ["plus", "multiplies", "min", "max", "bit_and", "bit_or", "bit_xor"],
)
def test_every_builtin_operator_remains_on_cudax(operator_name):
    operation = _reduce(
        operation="reduce",
        reduce_operator=_builtin_operator(operator_name),
    )
    plan = _plan(this_block(), operation)

    assert not operation.requests_cub
    assert plan.target is GroupLoweringTarget.CUDAX_GROUP


def test_reduce_result_is_always_one_scalar_of_the_payload_dtype():
    broadcast = _plan(
        this_block(),
        _reduce(dtype="float32", value_kind="array", items_per_thread=3),
    )
    root_only = _plan(
        this_block(),
        _reduce(dtype="float32", broadcast=False),
    )

    assert broadcast.result.primary.dtype == "float32"
    assert broadcast.result.primary.operand_kind is GroupOperandKind.SCALAR
    assert broadcast.result.primary.items_per_member == 1
    assert broadcast.result.primary.visibility is ResultVisibility.ALL_MEMBERS
    assert broadcast.result.primary.ownership is ResultOwnership.EACH_MEMBER
    assert broadcast.result.primary.root_rank is None
    assert root_only.result.primary.visibility is ResultVisibility.GROUP_ROOT
    assert root_only.result.primary.ownership is ResultOwnership.GROUP_ROOT
    assert root_only.result.primary.root_rank == 0


@pytest.mark.parametrize(
    ("group", "target", "scope", "instances"),
    [
        (
            this_block(),
            GroupLoweringTarget.CUB_BLOCK,
            SynchronizationScope.BLOCK,
            1,
        ),
        (
            this_warp(),
            GroupLoweringTarget.CUB_WARP,
            SynchronizationScope.WARP,
            2,
        ),
        (
            this_warp().group_by(8),
            GroupLoweringTarget.CUB_WARP,
            SynchronizationScope.WARP,
            8,
        ),
    ],
)
def test_valid_prefix_selects_root_only_cub_storage(
    group,
    target,
    scope,
    instances,
):
    plan = _plan(
        group,
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.runtime(),
        ),
    )

    assert plan.target is target
    assert plan.result.visibility is ResultVisibility.GROUP_ROOT
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space == "shared"
    assert plan.temp_storage.instances == instances
    assert plan.synchronization.storage_reuse_barrier is scope
    assert plan.participation.uniform_arguments == ("valid_items",)
    assert plan.participation.valid_member_selection == (
        "first N members by linear group rank"
    )
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (
        1,
        plan.resolved_group.static_size,
    )
    assert precondition.enforcement is PreconditionEnforcement.CALLER
    implementation_prefix = next(
        parameter
        for parameter in plan.implementation.parameters[0]
        if parameter.name in {"num_valid", "valid_items"}
    )
    assert implementation_prefix.dtype.name == "int32"


@pytest.mark.parametrize("valid_items", [0, -1, 33])
def test_static_warp_prefix_is_bounded_to_one_through_group_width(valid_items):
    if valid_items < 1:
        with pytest.raises(ValueError, match="positive integer"):
            _reduce(
                broadcast=False,
                valid_items=ArgumentBinding.static(valid_items),
            )
    else:
        operation = _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(valid_items),
        )
        with pytest.raises(ValueError, match="exceeds group size 32"):
            _plan(this_warp(), operation, 64)


def test_explicit_block_algorithm_and_array_payload_select_cub():
    plan = _plan(
        this_block(),
        _reduce(
            value_kind="array",
            items_per_thread=4,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 4
    assert plan.implementation.template_arguments["ALGORITHM"] == (
        "::cub::BLOCK_REDUCE_RAKING"
    )
    assert plan.result.primary.operand_kind is GroupOperandKind.SCALAR


def test_arbitrary_custom_operator_selects_cub_and_requires_root_only():
    custom = CxxOperator("custom_reduce<T>", Dependency("T"), name="binary_op")
    broadcast = _plan(
        this_block(),
        _reduce(operation="reduce", reduce_operator=custom),
    )
    root_only = _plan(
        this_warp(),
        _reduce(
            operation="reduce",
            reduce_operator=custom,
            broadcast=False,
        ),
    )

    assert broadcast.unsupported.code is UnsupportedReasonCode.CUB_BROADCAST
    assert root_only.target is GroupLoweringTarget.CUB_WARP


def test_block_reduce_algorithms_fail_closed_on_unproven_semantics():
    nondeterministic = _plan(
        this_block(),
        _reduce(
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
        ),
    )
    custom = CxxOperator("custom_reduce<T>", Dependency("T"), name="binary_op")
    unproven = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=custom,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    proven_sum = _plan(
        this_block(),
        _reduce(
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    proven_builtin = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=_builtin_operator("max"),
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )

    assert nondeterministic.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "addition-specific" in nondeterministic.unsupported.message
    assert unproven.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "proven commutativity" in unproven.unsupported.message
    assert proven_sum.target is GroupLoweringTarget.CUB_BLOCK
    assert proven_builtin.target is GroupLoweringTarget.CUB_BLOCK


def test_default_cub_algorithm_is_canonical_in_plan_identity():
    primitive = make_reduce_semantics(
        dtype="int32",
        items_per_thread=1,
        operation="sum",
        value_kind="scalar",
        valid_items=ArgumentBinding.static(np.int32(17)),
    )
    omitted = _plan(
        this_block(),
        GroupReduceSemantics(primitive, broadcast=False),
    )
    explicit = _plan(
        this_block(),
        GroupReduceSemantics(
            primitive,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        ),
    )

    assert omitted.call.operation.cub_algorithm is BlockReduceAlgorithm.WARP_REDUCTIONS
    assert omitted.semantic_key == explicit.semantic_key
    assert omitted.artifact_key == explicit.artifact_key


def test_custom_operator_and_prefix_are_declared_in_call_metadata():
    stateful = PythonOperator(
        ret_dtype=Dependency("T"),
        arg_dtypes=(Dependency("T"), Dependency("T")),
        op=lambda left, right: left + right,
        name="binary_op",
    )
    operation = _reduce(
        operation="reduce",
        reduce_operator=stateful,
        broadcast=False,
        valid_items=ArgumentBinding.static(7),
    )
    call = make_group_primitive_call(this_block(), operation)

    assert [item.name for item in call.argument_classifications] == [
        "value",
        "binary_op",
        "valid_items",
        "broadcast",
        "algorithm",
    ]
    assert [item.kind for item in call.argument_classifications] == [
        ArgumentKind.RUNTIME,
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
    ]
    assert call.argument_classifications[1].role is ParameterRole.OPERATOR
    assert call.argument_classifications[2].role is ParameterRole.CONSTANT


def test_nonexhaustive_mapped_topology_respects_physical_parent_boundaries():
    threads = _plan(
        this_warp().group_by(12, exhaustive=False),
        _reduce(),
        64,
    )
    warps = _plan(
        this_block().group_by(3, exhaustive=False),
        _reduce(),
        128,
    )

    assert threads.target is GroupLoweringTarget.CUDAX_GROUP
    assert threads.topology.instances == 4
    assert threads.topology.instance_index == (
        "(linear_thread_rank / 32) * 2 + ((linear_thread_rank % 32) / 12)"
    )
    assert threads.topology.thread_rank == "(linear_thread_rank % 32) % 12"
    assert not threads.participation.complete_parent_partition
    assert warps.target is GroupLoweringTarget.CUDAX_GROUP
    assert warps.topology.instances == 1
    assert warps.topology.instance_index == "(linear_thread_rank / 32) / 3"
    assert warps.topology.thread_rank == (
        "((linear_thread_rank / 32) % 3) * 32 + (linear_thread_rank % 32)"
    )
    assert not warps.participation.complete_parent_partition
    assert warps.temp_storage.ownership is StorageOwnership.NONE


def test_grid_reduce_has_a_stable_hidden_workspace_rejection():
    plan = _plan(this_grid(), _reduce(), LaunchFacts())

    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert plan.unsupported.message == (
        "cuda.coop Reduce does not support grid groups because grid reduction "
        "requires hidden per-launch workspace"
    )


class _ThreadData:
    items_per_thread = 2
    dtype = np.float32

    def __init__(self):
        self._items = [np.float32(1), np.float32(2)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def test_portable_reduce_matrix_and_family_owned_selectors(monkeypatch):
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
    with dispatch._compiler_scope("test.backend"):
        for group in groups:
            assert api.reduce(group, _ThreadData(), binary_op="+") is delegated
            assert api.sum(group, np.int32(1), broadcast=False) is delegated
        assert calls[0][1]["binary_op"] == "sum"
        with pytest.raises(NotImplementedError, match="hidden per-launch workspace"):
            api.sum(this_grid(), np.int32(1))
        with pytest.raises(TypeError, match="value dtypes"):
            api.reduce(this_block(), np.float32(1), binary_op="bit_and")
        with pytest.raises(ValueError, match="custom operators"):
            api.reduce(this_block(), np.int32(1), binary_op=object())


@pytest.mark.parametrize("operation", ["reduce", "sum"])
def test_portable_cub_controls_fail_closed_before_delegation(monkeypatch, operation):
    dispatch = import_module("cuda.coop._core.api._dispatch")
    api = import_module("cuda.coop._core.api.reduce")
    function = getattr(api, operation)
    calls = []
    monkeypatch.setattr(
        api,
        "_group_primitive_marker",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with dispatch._compiler_scope("test.backend"):
        with pytest.raises(ValueError, match="requires broadcast=False"):
            function(this_block(), np.int32(1), valid_items=17)
        with pytest.raises(ValueError, match="scalar values only"):
            function(
                this_block(),
                _ThreadData(),
                broadcast=False,
                valid_items=1,
            )
        with pytest.raises(ValueError, match="requires a block group"):
            function(
                this_warp(),
                np.int32(1),
                broadcast=False,
                algorithm="raking",
            )
        with pytest.raises(ValueError, match="at least 1"):
            function(
                this_block(),
                np.int32(1),
                broadcast=False,
                valid_items=0,
            )

    assert calls == []


def test_portable_root_exports_reduce_and_sum():
    import cuda.coop as coop

    api = import_module("cuda.coop._core.api.reduce")
    assert coop.reduce is api.reduce
    assert coop.sum is api.sum
    assert {"reduce", "sum"}.issubset(coop.__all__)

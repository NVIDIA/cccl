# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check reduction routing, result ownership, and group requirements.

All reductions use CUB and produce a result at group rank zero. Check
supported operand forms, valid prefixes, and temporary storage requirements
before a backend renders or compiles the calls.
"""

from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    BlockReduceAlgorithm,
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


_OMITTED_VALID_ITEMS = ArgumentBinding.omitted()


def _reduce(
    *,
    dtype="int32",
    operation="sum",
    value_kind="scalar",
    items_per_thread=1,
    reduce_operator=None,
    valid_items=_OMITTED_VALID_ITEMS,
    cub_algorithm=None,
    **storage,
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
        cub_algorithm=cub_algorithm,
        **storage,
    )


def _plan(group, operation, launch=64):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(
        make_group_primitive_call(group, operation), facts
    )


def _cluster_facts():
    """Supply verified cluster facts so tests reach reduction planning.

    The planner requires evidence of a cluster launch as well as its shape;
    shape values alone would fail before the reduction contract is checked.
    """

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
    ("group", "target", "instances"),
    [
        (this_block(), GroupLoweringTarget.CUB_BLOCK, 1),
        (this_warp(), GroupLoweringTarget.CUB_WARP, 2),
        (this_warp().group_by(8), GroupLoweringTarget.CUB_WARP, 8),
    ],
)
@pytest.mark.parametrize("operation", ["sum", "reduce"])
def test_full_reduce_uses_cub_and_compiler_managed_scratch(
    group, target, instances, operation
):
    plan = _plan(group, _reduce(operation=operation))

    assert plan.target is target
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space == "shared"
    assert plan.temp_storage.instances == instances
    assert plan.temp_storage.auto_sync
    assert plan.provenance.library == "CUB"


@pytest.mark.parametrize(
    "operator_name",
    ["plus", "multiplies", "min", "max", "bit_and", "bit_or", "bit_xor"],
)
def test_every_builtin_operator_uses_cub(operator_name):
    plan = _plan(
        this_block(),
        _reduce(
            operation="reduce", reduce_operator=_builtin_operator(operator_name)
        ),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_block_array_result_is_one_scalar_defined_at_rank_zero(
    items_per_thread,
):
    plan = _plan(
        this_block(),
        _reduce(
            dtype="float32",
            value_kind="array",
            items_per_thread=items_per_thread,
        ),
    )

    assert plan.result.primary.dtype == "float32"
    assert plan.result.primary.operand_kind is GroupOperandKind.SCALAR
    assert plan.result.primary.items_per_member == 1
    assert plan.result.primary.visibility is ResultVisibility.GROUP_ROOT
    assert plan.result.primary.ownership is ResultOwnership.GROUP_ROOT
    assert plan.result.primary.root_rank == 0


@pytest.mark.parametrize("auto_sync", [False, True])
@pytest.mark.parametrize("sharing", ["shared", "exclusive"])
def test_explicit_block_storage_preserves_layout_and_reuse_policy(
    auto_sync, sharing
):
    plan = _plan(
        this_block(),
        _reduce(
            storage_ownership=StorageOwnership.CALLER,
            storage_sharing=sharing,
            storage_size_in_bytes=512,
            storage_alignment=16,
            storage_auto_sync=auto_sync,
        ),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.temp_storage.ownership is StorageOwnership.CALLER
    assert plan.temp_storage.exact_layout_required
    assert plan.temp_storage.sharing == sharing
    assert plan.temp_storage.requested_size_in_bytes == 512
    assert plan.temp_storage.requested_alignment == 16
    assert plan.temp_storage.auto_sync is auto_sync
    assert plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


@pytest.mark.parametrize("group", [this_warp(), this_warp().group_by(8)])
def test_warp_rejects_explicit_storage(group):
    explicit_storage = _plan(
        group,
        _reduce(
            storage_ownership=StorageOwnership.CALLER, storage_sharing="shared"
        ),
    )

    assert (
        explicit_storage.unsupported.code
        is UnsupportedReasonCode.OPERATION_VARIANT
    )
    assert "temp_storage" in explicit_storage.unsupported.message


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
                valid_items=ArgumentBinding.static(valid_items),
            )
    else:
        operation = _reduce(
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
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 4
    assert plan.implementation.template_arguments["ALGORITHM"] == (
        "::cub::BLOCK_REDUCE_RAKING"
    )
    assert plan.result.primary.operand_kind is GroupOperandKind.SCALAR


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_custom_operator_uses_cub_and_a_root_only_result(group):
    custom = CxxOperator("custom_reduce<T>", Dependency("T"), name="binary_op")
    plan = _plan(group, _reduce(operation="reduce", reduce_operator=custom))

    assert plan.provenance.library == "CUB"
    assert plan.result.primary.visibility is ResultVisibility.GROUP_ROOT


def test_block_reduce_algorithms_fail_closed_on_unproven_semantics():
    nondeterministic = _plan(
        this_block(),
        _reduce(
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
        ),
    )
    custom = CxxOperator("custom_reduce<T>", Dependency("T"), name="binary_op")
    unproven = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=custom,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    proven_sum = _plan(
        this_block(),
        _reduce(
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    proven_builtin = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=_builtin_operator("max"),
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )

    assert (
        nondeterministic.unsupported.code
        is UnsupportedReasonCode.OPERATION_VARIANT
    )
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
        GroupReduceSemantics(primitive),
    )
    explicit = _plan(
        this_block(),
        GroupReduceSemantics(
            primitive,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        ),
    )

    assert (
        omitted.call.operation.cub_algorithm
        is BlockReduceAlgorithm.WARP_REDUCTIONS
    )
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
        valid_items=ArgumentBinding.static(7),
    )
    call = make_group_primitive_call(this_block(), operation)

    assert [item.name for item in call.argument_classifications] == [
        "value",
        "binary_op",
        "valid_items",
        "algorithm",
    ]
    assert [item.kind for item in call.argument_classifications] == [
        ArgumentKind.RUNTIME,
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
    ]
    assert call.argument_classifications[1].role is ParameterRole.OPERATOR
    assert call.argument_classifications[2].role is ParameterRole.CONSTANT


@pytest.mark.parametrize(
    ("group", "facts"),
    [
        (this_thread(), LaunchFacts(64)),
        (this_block().group_by(2), LaunchFacts(128)),
        (this_cluster(), _cluster_facts()),
        (this_grid(), LaunchFacts()),
    ],
)
def test_reduction_rejects_groups_without_a_cub_implementation(group, facts):
    plan = _plan(group, _reduce(), facts)

    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND


@pytest.mark.parametrize("width", [17, 31])
def test_nonexhaustive_logical_warp_resets_scratch_instances_per_warp(width):
    plan = _plan(this_warp().group_by(width, exhaustive=False), _reduce(), 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.topology.instances == 2


def test_nonexhaustive_warp_rejects_multiple_non_power_of_two_groups():
    plan = _plan(this_warp().group_by(12, exhaustive=False), _reduce(), 64)

    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert "only one non-power-of-two group" in plan.unsupported.message


def test_complete_nonexhaustive_logical_warp_uses_canonical_cub_topology():
    plan = _plan(
        this_warp().group_by(8, exhaustive=False),
        _reduce(
            valid_items=ArgumentBinding.runtime(),
        ),
        64,
    )

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.resolved_group.complete_membership is True
    assert plan.topology.instances == 8
    assert plan.topology.instance_index == "linear_thread_rank / 8"
    assert plan.topology.thread_rank == "linear_thread_rank % 8"
    assert plan.temp_storage.instances == 8
    assert plan.temp_storage.instance_index == "linear_thread_rank / 8"


def test_common_root_exports_reduce_and_sum():
    from cuda import coop

    api = import_module("cuda.coop._core.api.reduce")
    assert coop.reduce is api.reduce
    assert coop.sum is api.sum
    assert {"reduce", "sum"}.issubset(coop.__all__)

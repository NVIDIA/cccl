# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group reduce planner contracts."""

import pytest

from tests.support.group_planning import (
    AlgorithmSpec,
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
    ResultOwnership,
    ResultVisibility,
    StatefulOperator,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    _plan,
    _reduce,
    _scan,
    make_block_reduce_semantics,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)


@pytest.mark.parametrize(
    ("group", "barrier"),
    [
        (this_block(), SynchronizationScope.BLOCK),
        (this_warp(), SynchronizationScope.WARP),
    ],
)
def test_default_reduce_selects_broadcasted_cudax(group, barrier):
    operation = _reduce(
        operand_kind=GroupOperandKind.ARRAY,
        items_per_thread=4,
    )
    plan = _plan(group, operation, 128)

    assert plan.target is GroupLoweringTarget.CUDAX_GROUP
    assert plan.implementation.primitive == "reduce"
    assert plan.implementation.overload == "broadcasted"
    assert plan.implementation.return_kind is CudaxReturnKind.VALUE
    assert [parameter.name for parameter in plan.implementation.parameters] == [
        "item0",
        "item1",
        "item2",
        "item3",
    ]
    assert plan.provenance.library == "CUDAX"
    assert plan.provenance.header == "cuda/experimental/coop.cuh"
    assert plan.result.visibility is ResultVisibility.ALL_MEMBERS
    assert plan.result.primary.ownership is ResultOwnership.EACH_MEMBER
    assert plan.result.primary.root_rank is None
    assert plan.result.operand_kind is GroupOperandKind.SCALAR
    assert plan.result.result_items_per_thread == 1
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.instances is None
    assert plan.synchronization.storage_reuse_barrier is barrier


@pytest.mark.parametrize(
    ("group", "target", "class_name", "instances", "barrier"),
    [
        (this_block(), GroupLoweringTarget.CUB_BLOCK, "cub::BlockReduce", 1, "block"),
        (this_warp(), GroupLoweringTarget.CUB_WARP, "cub::WarpReduce", 4, "warp"),
    ],
)
def test_cub_only_reduce_is_root_only_with_exact_storage(
    group,
    target,
    class_name,
    instances,
    barrier,
):
    operation = _reduce(
        broadcast=False,
        valid_items=ArgumentBinding.static(17),
    )
    call = make_group_primitive_call(group, operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=128))

    assert plan.target is target
    assert isinstance(plan.implementation, AlgorithmSpec)
    assert plan.provenance.cpp_class == class_name
    assert plan.result.visibility is ResultVisibility.GROUP_ROOT
    assert plan.result.primary.ownership is ResultOwnership.GROUP_ROOT
    assert plan.result.primary.root_rank == 0
    assert plan.temp_storage.ownership is StorageOwnership.CALLER
    assert plan.temp_storage.instances == instances
    assert plan.temp_storage.cpp_type == "typename implementation_type::TempStorage"
    assert plan.synchronization.storage_reuse_barrier.value == barrier
    assert call.argument_classifications[1].kind is ArgumentKind.STATIC
    assert call.argument_classifications[1].role is ParameterRole.CONSTANT
    implementation_classifications = plan.implementation.classify_method()
    implementation_valid_items = next(
        item
        for item in implementation_classifications
        if item.name in {"num_valid", "valid_items"}
    )
    assert implementation_valid_items.kind is ArgumentKind.STATIC
    assert implementation_valid_items.role is ParameterRole.CONSTANT
    assert plan.participation.uniform_arguments == ("valid_items",)
    assert plan.participation.valid_member_selection.startswith("first N members")
    precondition = plan.participation.argument_preconditions[0]
    assert precondition.name == "valid_items"
    assert (precondition.minimum, precondition.maximum) == (
        1,
        plan.resolved_group.static_size,
    )
    assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED


def test_partial_logical_warp_reduce_uses_mapped_width_and_storage_instances():
    plan = _plan(
        this_warp().group_by(8),
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(5),
        ),
        64,
    )

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 8
    assert plan.implementation.method_name == "Sum"
    assert plan.temp_storage.instances == 8
    assert plan.temp_storage.instance_index == "linear_thread_rank / 8"
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (1, 8)


def test_cub_reduce_cannot_synthesize_broadcast_and_warp_has_no_algorithm_tag():
    broadcast = _plan(
        this_block(),
        _reduce(valid_items=ArgumentBinding.runtime()),
    )
    warp_algorithm = _plan(
        this_warp(),
        _reduce(
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
    )

    assert broadcast.unsupported.code is UnsupportedReasonCode.CUB_BROADCAST
    assert warp_algorithm.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT

    runtime_call = make_group_primitive_call(
        this_block(),
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.runtime(),
        ),
    )
    assert runtime_call.argument_classifications[1].kind is ArgumentKind.RUNTIME
    assert runtime_call.argument_classifications[1].role is ParameterRole.INPUT

    with pytest.raises(ValueError, match="unsupported CUB BlockReduce algorithm"):
        _reduce(broadcast=False, cub_algorithm="::cub::BLOCK_SCAN_RAKING")
    with pytest.raises(ValueError, match="positive integer"):
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(0),
        )
    with pytest.raises(ValueError, match="exceeds group size"):
        _plan(
            this_warp(),
            _reduce(
                broadcast=False,
                valid_items=ArgumentBinding.static(33),
            ),
            64,
        )


def test_group_block_reduce_algorithms_fail_closed_on_unproven_semantics():
    nondeterministic = _plan(
        this_block(),
        _reduce(
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
        ),
    )
    assert nondeterministic.target is GroupLoweringTarget.UNSUPPORTED
    assert nondeterministic.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "addition-specific" in nondeterministic.unsupported.message

    unproven_commutative = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=CxxOperator("custom_reduce<T>", Dependency("T")),
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    assert unproven_commutative.target is GroupLoweringTarget.UNSUPPORTED
    assert "proven commutativity" in unproven_commutative.unsupported.message

    proven_commutative = _plan(
        this_block(),
        _reduce(
            operation="max",
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    assert proven_commutative.target is GroupLoweringTarget.CUB_BLOCK


def test_runtime_valid_items_is_an_explicit_caller_precondition():
    plan = _plan(
        this_block(),
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.runtime(),
        ),
        64,
    )

    precondition = plan.participation.argument_preconditions[0]
    assert precondition.name == "valid_items"
    assert (precondition.minimum, precondition.maximum) == (1, 64)
    assert precondition.enforcement is PreconditionEnforcement.CALLER
    precondition.validate(1)
    precondition.validate(64)
    for invalid in (-1, 0, 65):
        with pytest.raises(ValueError, match="valid_items must be"):
            precondition.validate(invalid)


@pytest.mark.parametrize(
    ("group", "valid_items", "target"),
    [
        (this_block(), 128, GroupLoweringTarget.CUB_BLOCK),
        (this_warp(), 32, GroupLoweringTarget.CUB_WARP),
    ],
)
def test_full_size_valid_items_still_selects_root_only_cub(
    group,
    valid_items,
    target,
):
    plan = _plan(
        group,
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(valid_items),
        ),
        128,
    )

    assert plan.target is target
    assert plan.result.visibility is ResultVisibility.GROUP_ROOT
    assert plan.participation.valid_member_selection == (
        "first N members by linear group rank"
    )
    precondition = plan.participation.argument_preconditions[0]
    assert precondition.maximum == valid_items
    assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED


def test_stateful_operator_state_is_uniform_and_remains_in_the_runtime_abi():
    stateful = StatefulOperator(
        op=lambda left, right: left + right,
        state_dtype="state",
        ret_dtype="int",
        arg_dtypes=("int", "int"),
    )
    cudax = _plan(
        this_block(),
        _reduce(operation="reduce", reduce_operator=stateful),
        64,
    )
    direct_cub = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=stateful,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
        64,
    )
    scan = _plan(
        this_warp(),
        _scan(mode="inclusive", scan_operator=stateful),
        64,
    )

    assert cudax.participation.uniform_arguments == ("operation",)
    assert [parameter.name for parameter in cudax.implementation.parameters] == [
        "item0",
        "operation",
    ]
    assert direct_cub.participation.uniform_arguments == ("operation",)
    assert scan.participation.uniform_arguments == ("operation",)


def test_reduce_plans_every_static_cudax_group_form():
    cluster_facts = LaunchFacts(
        exact_block_dim=64,
        exact_cluster_dim=2,
        cluster_launch=True,
        provenance=(
            LaunchFactOrigin("exact_cluster_dim", "launch_config", verified=True),
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )
    grid_facts = LaunchFacts(
        exact_block_dim=64,
        exact_grid_dim=8,
        exact_cluster_dim=2,
        cluster_launch=True,
        cooperative_launch=True,
        provenance=(
            LaunchFactOrigin("exact_grid_dim", "launch_config", verified=True),
            LaunchFactOrigin("exact_cluster_dim", "launch_config", verified=True),
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
            LaunchFactOrigin("cooperative_launch", "launch_config", verified=True),
        ),
    )
    cases = (
        (this_thread(), LaunchFacts()),
        (this_warp(), LaunchFacts(exact_block_dim=64)),
        (this_block(), LaunchFacts(exact_block_dim=64)),
        (
            this_cluster(),
            LaunchFacts(
                exact_block_dim=64,
                cluster_launch=False,
                provenance=(
                    LaunchFactOrigin(
                        "cluster_launch",
                        "launch_config",
                        verified=True,
                    ),
                ),
            ),
        ),
        (this_cluster(), cluster_facts),
        (this_grid(), grid_facts),
        (
            this_warp().group_by(12, exhaustive=False),
            LaunchFacts(exact_block_dim=64),
        ),
        (
            this_block().group_by(3, exhaustive=False),
            LaunchFacts(exact_block_dim=320),
        ),
    )

    plans = [_plan(group, _reduce(), facts) for group, facts in cases]
    assert all(plan.target is GroupLoweringTarget.CUDAX_GROUP for plan in plans)
    assert plans[0].participation.exact_block_dim is None
    assert plans[3].resolved_group.hierarchy.cluster_dim == (1, 1, 1)
    assert plans[4].resolved_group.static_size == 128
    assert plans[5].resolved_group.hierarchy.grid_dim == (4, 1, 1)
    assert plans[5].resolved_group.static_size == 512
    assert plans[6].participation.complete_membership is False
    assert plans[7].resolved_group.groups_per_parent == 3


def test_explicit_cub_algorithm_participates_in_call_and_plan_identity():
    primitive = make_block_reduce_semantics(
        dtype="int",
        operation="sum",
        value_kind="scalar",
        items_per_thread=1,
    )
    raking_call = make_group_primitive_call(
        this_block(),
        GroupReduceSemantics(
            primitive,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
    )
    warp_call = make_group_primitive_call(
        this_block(),
        GroupReduceSemantics(
            primitive,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        ),
    )
    raking_plan = plan_group_primitive(raking_call, LaunchFacts(exact_block_dim=64))
    warp_plan = plan_group_primitive(warp_call, LaunchFacts(exact_block_dim=64))

    assert raking_call.semantic_key != warp_call.semantic_key
    assert raking_plan.semantic_key != warp_plan.semantic_key
    assert raking_plan.artifact_key != warp_plan.artifact_key


def test_omitted_block_cub_algorithm_canonicalizes_to_warp_reductions():
    primitive = make_block_reduce_semantics(
        dtype="int",
        operation="sum",
        value_kind="scalar",
        items_per_thread=1,
        valid_items=ArgumentBinding.static(17),
    )
    omitted = plan_group_primitive(
        make_group_primitive_call(
            this_block(),
            GroupReduceSemantics(primitive, broadcast=False),
        ),
        LaunchFacts(exact_block_dim=64),
    )
    explicit = plan_group_primitive(
        make_group_primitive_call(
            this_block(),
            GroupReduceSemantics(
                primitive,
                broadcast=False,
                cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
            ),
        ),
        LaunchFacts(exact_block_dim=64),
    )

    assert omitted.call == explicit.call
    assert omitted.semantic_key == explicit.semantic_key
    assert omitted.artifact_key == explicit.artifact_key
    assert omitted == explicit

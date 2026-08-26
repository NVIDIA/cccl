# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group model planner contracts."""

import pytest

from tests.support.group_planning import (
    ArgumentBinding,
    CudaxReturnKind,
    GroupExchangeSemantics,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupReduceSemantics,
    GroupScanSemantics,
    LaunchFactOrigin,
    LaunchFacts,
    LogicalResultContract,
    ResultOwnership,
    ResultVisibility,
    ThreadGroup,
    ThreadHierarchy,
    UnsupportedReasonCode,
    _exchange,
    _load_store,
    _plan,
    _reduce,
    _scan,
    make_block_exchange_semantics,
    make_block_exchange_spec,
    make_block_reduce_semantics,
    make_block_scan_spec,
    make_group_primitive_call,
    make_scan_semantics,
    make_warp_exchange_spec,
    make_warp_reduce_spec,
    make_warp_scan_spec,
    plan_group_primitive,
    this_block,
    this_warp,
)


def test_logical_result_contract_enforces_group_root_rank_zero():
    valid = LogicalResultContract(
        name="value",
        dtype="int",
        visibility=ResultVisibility.GROUP_ROOT,
        ownership=ResultOwnership.GROUP_ROOT,
        operand_kind=GroupOperandKind.SCALAR,
        items_per_member=1,
        root_rank=0,
    )

    assert valid.root_rank == 0
    for invalid_rank in (None, -1, 1, 999, True):
        with pytest.raises(ValueError, match="root rank 0"):
            LogicalResultContract(
                name="value",
                dtype="int",
                visibility=ResultVisibility.GROUP_ROOT,
                ownership=ResultOwnership.GROUP_ROOT,
                operand_kind=GroupOperandKind.SCALAR,
                items_per_member=1,
                root_rank=invalid_rank,
            )
    with pytest.raises(ValueError, match="cannot define a root rank"):
        LogicalResultContract(
            name="value",
            dtype="int",
            visibility=ResultVisibility.ALL_MEMBERS,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=GroupOperandKind.SCALAR,
            items_per_member=1,
            root_rank=0,
        )


def test_warp_semantics_ignore_parent_cta_but_artifacts_do_not():
    operation = _reduce()
    first = _plan(this_warp(), operation, 64)
    second = _plan(
        ThreadGroup(
            kind="warp",
            hierarchy=ThreadHierarchy._resolved(
                block_dim=128,
                grid_dim=7,
                cluster_dim=2,
            ),
            source="resolved_with_irrelevant_facts",
        ),
        operation,
        128,
    )

    assert first.semantic_key == second.semantic_key
    assert first.artifact_key != second.artifact_key
    assert first.participation.exact_group_size == 32
    assert second.participation.exact_group_size == 32
    assert first.participation.complete_parent_partition


def test_diagnostic_sources_do_not_fragment_plan_or_artifact_identity():
    operation = _reduce()
    first = plan_group_primitive(
        make_group_primitive_call(
            ThreadGroup(
                kind="block",
                source="root_frontend",
            ),
            operation,
            source="root_frontend",
        ),
        LaunchFacts(
            exact_block_dim=64,
            provenance=LaunchFactOrigin("exact_block_dim", "call_metadata"),
        ),
    )
    second = plan_group_primitive(
        make_group_primitive_call(
            ThreadGroup(
                kind="block",
                source="scoped_frontend",
            ),
            operation,
            source="scoped_frontend",
        ),
        LaunchFacts(
            exact_block_dim=64,
            provenance=LaunchFactOrigin("exact_block_dim", "reqntid"),
        ),
    )

    assert first.semantic_key == second.semantic_key
    assert first.artifact_key == second.artifact_key
    assert first == second


def test_group_requests_reuse_scoped_core_semantics_exactly():
    reduce_primitive = make_block_reduce_semantics(
        dtype="int",
        operation="sum",
        value_kind="scalar",
        items_per_thread=1,
    )
    reduce_operation = GroupReduceSemantics(reduce_primitive)
    exchange_primitive = make_block_exchange_semantics(
        dtype="int",
        mode="striped_to_blocked",
        items_per_thread=2,
    )
    exchange_operation = GroupExchangeSemantics(exchange_primitive)
    scan_primitive = make_scan_semantics(
        dtype="int",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
    )
    scan_operation = GroupScanSemantics(scan_primitive)

    block_scan = make_block_scan_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=1,
        mode="exclusive",
        algorithm="::cub::BLOCK_SCAN_RAKING",
        value_kind="scalar",
    )
    warp_scan = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=32,
        mode="exclusive",
    )
    warp_reduce = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=32,
        operation="sum",
    )
    warp_exchange = make_warp_exchange_spec(
        dtype="int",
        items_per_thread=2,
        threads_in_warp=32,
        mode="striped_to_blocked",
    )
    block_exchange = make_block_exchange_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=2,
        mode="striped_to_blocked",
        value_form="out_of_place",
        warp_time_slicing=False,
    )

    assert reduce_operation.primitive is reduce_primitive
    assert exchange_operation.primitive is exchange_primitive
    assert scan_operation.primitive is scan_primitive
    assert warp_reduce.call.semantic_key == reduce_primitive.semantic_key
    assert warp_exchange.call.semantic_key == exchange_primitive.semantic_key
    assert block_scan.call.semantic_key == scan_primitive.semantic_key
    assert warp_scan.call.semantic_key == scan_primitive.semantic_key
    assert _plan(this_block(), reduce_operation).semantic_key[1][0] == (
        reduce_primitive.semantic_key
    )
    assert _plan(this_warp(), reduce_operation).semantic_key[1][0] == (
        reduce_primitive.semantic_key
    )
    assert _plan(this_block(), exchange_operation).semantic_key[1] == (
        exchange_primitive.semantic_key
    )
    assert _plan(this_block(), scan_operation).semantic_key[1][0] == (
        scan_primitive.semantic_key
    )
    assert _plan(this_block(), scan_operation).implementation == (
        block_scan.specialization
    )
    assert _plan(this_warp(), scan_operation).implementation == (
        warp_scan.specialization
    )
    assert _plan(this_block(), exchange_operation).implementation == (
        block_exchange.specialization
    )
    assert _plan(this_warp(), exchange_operation).implementation == (
        warp_exchange.specialization
    )


@pytest.mark.parametrize(
    "operation",
    [
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(5),
        ),
        _scan(mode="inclusive"),
        _exchange(),
        _load_store("load"),
        _load_store("store"),
    ],
)
def test_cub_backed_logical_warp_rejects_non_power_of_two_width(operation):
    group = this_warp().group_by(12, exhaustive=False)
    plan = _plan(group, operation, 96)

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.implementation is None
    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert "power-of-two group width" in plan.unsupported.message
    assert "got 12" in plan.unsupported.message


@pytest.mark.parametrize(
    ("operation", "template_parameter"),
    [
        (
            _reduce(
                broadcast=False,
                valid_items=ArgumentBinding.static(5),
            ),
            "VIRTUAL_WARP_THREADS",
        ),
        (_scan(mode="inclusive"), "VIRTUAL_WARP_THREADS"),
        (_exchange(), "LOGICAL_WARP_THREADS"),
        (_load_store("load"), "LOGICAL_WARP_THREADS"),
        (_load_store("store"), "LOGICAL_WARP_THREADS"),
    ],
)
def test_physical_warp_cub_plans_use_architectural_width(
    operation,
    template_parameter,
):
    plan = _plan(this_warp(), operation, 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.template_arguments[template_parameter] == 32


def test_group_and_launch_markers_are_erased_from_runtime_abi():
    operation = _exchange("striped_to_blocked", 2)
    call = make_group_primitive_call(
        this_block(),
        operation,
        source="root_frontend",
    )
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert all(
        classification.name not in {"group", "launch", "launch_facts"}
        for classification in call.argument_classifications
    )
    assert all(
        not isinstance(parameter, (ThreadGroup, LaunchFacts))
        for method in plan.implementation.parameters
        for parameter in method
    )

    cudax_plan = _plan(this_warp(), _reduce(broadcast=False), 64)
    assert cudax_plan.implementation.return_kind is CudaxReturnKind.OPTIONAL_VALUE
    assert all(
        parameter.name not in {"group", "launch", "launch_facts"}
        for parameter in cudax_plan.implementation.parameters
    )


def test_operation_semantics_participate_in_artifact_identity():
    sum_plan = _plan(this_block(), _reduce(operation="sum"), 64)
    max_plan = _plan(this_block(), _reduce(operation="max"), 64)

    assert sum_plan.artifact_key != max_plan.artifact_key
    assert sum_plan != max_plan

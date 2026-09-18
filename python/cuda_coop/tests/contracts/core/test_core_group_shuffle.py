# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupShuffleSemantics,
    LaunchFacts,
    ParameterRole,
    PreconditionEnforcement,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    make_block_shuffle_semantics,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_warp,
)


def _shuffle(mode="offset", **overrides):
    primitive = make_block_shuffle_semantics(
        dtype=overrides.pop("dtype", "i32"),
        mode=mode,
        items_per_thread=overrides.pop("items_per_thread", None),
        distance=overrides.pop("distance", None),
    )
    assert not overrides
    return GroupShuffleSemantics(primitive)


def _plan(group, operation, launch=(64, 1, 1)):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(make_group_primitive_call(group, operation), facts)


@pytest.mark.parametrize(
    ("operation", "method", "operand_kind", "items_per_member"),
    [
        (_shuffle("offset"), "Offset", GroupOperandKind.SCALAR, 1),
        (_shuffle("rotate"), "Rotate", GroupOperandKind.SCALAR, 1),
        (
            _shuffle("up", items_per_thread=3),
            "Up",
            GroupOperandKind.ARRAY,
            3,
        ),
        (
            _shuffle("down", items_per_thread=3),
            "Down",
            GroupOperandKind.ARRAY,
            3,
        ),
    ],
)
def test_group_shuffle_has_explicit_result_topology_and_storage_contracts(
    operation,
    method,
    operand_kind,
    items_per_member,
):
    plan = _plan(this_block(), operation, (8, 4, 2))

    assert operation.result_visibility is ResultVisibility.PER_MEMBER
    assert operation.returns_value
    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.method_name == method
    assert plan.result.values[0].name == "value"
    assert plan.result.values[0].dtype == "i32"
    assert plan.result.values[0].visibility is ResultVisibility.PER_MEMBER
    assert plan.result.values[0].ownership is ResultOwnership.EACH_MEMBER
    assert plan.result.values[0].operand_kind is operand_kind
    assert plan.result.values[0].items_per_member == items_per_member
    assert plan.topology.group_kind == "block"
    assert plan.topology.logical_width == 64
    assert plan.topology.instances == 1
    assert plan.topology.instance_index == "cta"
    assert plan.topology.thread_rank == "linear_thread_rank"
    assert plan.topology.execution_scope is SynchronizationScope.BLOCK
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space == "shared"
    assert plan.temp_storage.instances == 1
    assert plan.temp_storage.instance_index == "cta"
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert plan.provenance.header == "cub/block/block_shuffle.cuh"


@pytest.mark.parametrize(
    ("distance", "kind", "role"),
    [
        (ArgumentBinding.runtime(), ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentBinding.static(3), ArgumentKind.STATIC, ParameterRole.CONSTANT),
    ],
)
def test_group_shuffle_classifies_distance_without_claiming_uniformity(
    distance,
    kind,
    role,
):
    operation = _shuffle("rotate", distance=distance)
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(64))

    assert [item.name for item in call.argument_classifications] == [
        "value",
        "distance",
        "mode",
    ]
    assert call.argument_classifications[1].kind is kind
    assert call.argument_classifications[1].role is role
    assert plan.participation.uniform_arguments == ()
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.name, precondition.minimum, precondition.maximum) == (
        "distance",
        1,
        63,
    )
    assert precondition.enforcement is (
        PreconditionEnforcement.CALLER
        if kind is ArgumentKind.RUNTIME
        else PreconditionEnforcement.PLANNER_VALIDATED
    )


def test_group_shuffle_default_distance_has_no_runtime_binding():
    call = make_group_primitive_call(this_block(), _shuffle("offset"))
    plan = plan_group_primitive(call, LaunchFacts(64))

    assert [item.name for item in call.argument_classifications] == [
        "value",
        "mode",
    ]
    assert plan.implementation.parameters[0][-1].cpp == "1"


def test_group_shuffle_preserves_input_and_cache_relevant_policy():
    first = _plan(this_block(), _shuffle("rotate", distance=ArgumentBinding.runtime()))
    equivalent = _plan(
        this_block(),
        _shuffle("rotate", distance=ArgumentBinding.runtime()),
    )
    static = _plan(
        this_block(),
        _shuffle("rotate", distance=ArgumentBinding.static(2)),
    )
    different_dtype = _plan(this_block(), _shuffle("rotate", dtype="i64"))
    array = _plan(this_block(), _shuffle("up", items_per_thread=2))

    assert first.semantic_key == equivalent.semantic_key
    assert first.artifact_key == equivalent.artifact_key
    assert first.artifact_key != static.artifact_key
    assert first.artifact_key != different_dtype.artifact_key
    assert first.artifact_key != array.artifact_key
    assert first.implementation.parameters[0][1].name == "input_item"
    assert not first.implementation.parameters[0][1].is_inout
    assert first.implementation.parameters[0][2].is_output


def test_group_shuffle_rejects_non_block_groups_and_non_public_shapes():
    warp = _plan(this_warp(), _shuffle("offset"))
    scalar_up = _plan(this_block(), _shuffle("up"))
    array_rotate = _plan(this_block(), _shuffle("rotate", items_per_thread=2))
    shifted_array = _plan(
        this_block(),
        _shuffle(
            "down",
            items_per_thread=2,
            distance=ArgumentBinding.static(2),
        ),
    )
    missing_shape = _plan(
        this_block(),
        _shuffle("offset"),
        LaunchFacts(max_block_dim=64),
    )
    one_thread_rotate = _plan(this_block(), _shuffle("rotate"), 1)

    assert warp.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert scalar_up.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert array_rotate.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert shifted_array.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert missing_shape.unsupported.code is (
        UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    )
    assert one_thread_rotate.unsupported.code is (
        UnsupportedReasonCode.OPERATION_VARIANT
    )

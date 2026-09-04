# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core import (
    ArgumentKind,
    GroupExchangeSemantics,
    GroupLoweringTarget,
    GroupOperandKind,
    LaunchFacts,
    ParameterRole,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    make_block_exchange_semantics,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_grid,
    this_warp,
)


def _exchange(mode="striped_to_blocked", **overrides):
    uses_ranks = mode.startswith("scatter_")
    uses_flags = mode == "scatter_to_striped_flagged"
    primitive = make_block_exchange_semantics(
        dtype=overrides.pop("dtype", "i32"),
        items_per_thread=overrides.pop("items_per_thread", 3),
        mode=mode,
        value_form=overrides.pop("value_form", "out_of_place"),
        warp_time_slicing=overrides.pop("warp_time_slicing", False),
        rank_dtype=overrides.pop("rank_dtype", "i32" if uses_ranks else None),
        valid_flag_dtype=overrides.pop(
            "valid_flag_dtype",
            "u8" if uses_flags else None,
        ),
    )
    assert not overrides
    return GroupExchangeSemantics(primitive)


def _plan(group, operation, launch=64):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(make_group_primitive_call(group, operation), facts)


@pytest.mark.parametrize(
    ("group", "target", "scope", "instances", "instance_index"),
    [
        (
            this_block(),
            GroupLoweringTarget.CUB_BLOCK,
            SynchronizationScope.BLOCK,
            1,
            "cta",
        ),
        (
            this_warp(),
            GroupLoweringTarget.CUB_WARP,
            SynchronizationScope.WARP,
            2,
            "linear_thread_rank / 32",
        ),
    ],
)
@pytest.mark.parametrize(
    ("mode", "method"),
    [
        ("striped_to_blocked", "StripedToBlocked"),
        ("blocked_to_striped", "BlockedToStriped"),
    ],
)
def test_group_exchange_has_explicit_result_topology_and_storage_contracts(
    group,
    target,
    scope,
    instances,
    instance_index,
    mode,
    method,
):
    operation = _exchange(mode)
    plan = _plan(group, operation)

    assert operation.result_visibility is ResultVisibility.PER_MEMBER
    assert operation.returns_value
    assert plan.target is target
    assert plan.implementation.method_name == method
    assert plan.result.values[0].name == "value"
    assert plan.result.values[0].dtype == "i32"
    assert plan.result.values[0].visibility is ResultVisibility.PER_MEMBER
    assert plan.result.values[0].ownership is ResultOwnership.EACH_MEMBER
    assert plan.result.values[0].operand_kind is GroupOperandKind.ARRAY
    assert plan.result.values[0].items_per_member == 3
    assert plan.topology.instances == instances
    assert plan.topology.instance_index == instance_index
    assert plan.topology.execution_scope is scope
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space == "shared"
    assert plan.temp_storage.instances == instances
    assert plan.temp_storage.instance_index == instance_index
    assert plan.synchronization.storage_reuse_barrier is scope


@pytest.mark.parametrize("logical_width", [1, 2, 4, 8, 16, 32])
def test_logical_warp_exchange_uses_every_group_instance(logical_width):
    plan = _plan(this_warp().group_by(logical_width), _exchange(), 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == (
        logical_width
    )
    assert plan.topology.group_kind == "threads_within_warp"
    assert plan.topology.logical_width == logical_width
    assert plan.topology.instances == 64 // logical_width
    assert plan.topology.instance_index == f"linear_thread_rank / {logical_width}"
    assert plan.topology.thread_rank == f"linear_thread_rank % {logical_width}"
    assert plan.temp_storage.instances == 64 // logical_width
    assert plan.temp_storage.instance_index == (f"linear_thread_rank / {logical_width}")
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP


@pytest.mark.parametrize(
    ("mode", "method", "time_slicing"),
    [
        ("striped_to_blocked", "StripedToBlocked", True),
        ("blocked_to_striped", "BlockedToStriped", True),
        ("warp_striped_to_blocked", "WarpStripedToBlocked", True),
        ("blocked_to_warp_striped", "BlockedToWarpStriped", True),
        ("scatter_to_blocked", "ScatterToBlocked", True),
        ("scatter_to_striped", "ScatterToStriped", True),
        ("scatter_to_striped_guarded", "ScatterToStripedGuarded", False),
        ("scatter_to_striped_flagged", "ScatterToStripedFlagged", False),
    ],
)
def test_group_exchange_plans_every_block_mode(mode, method, time_slicing):
    operation = _exchange(mode, warp_time_slicing=time_slicing)
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(64))

    expected_runtime = ["value"]
    if operation.primitive.uses_ranks:
        expected_runtime.append("ranks")
    if operation.primitive.uses_valid_flags:
        expected_runtime.append("valid_flags")
    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.method_name == method
    assert plan.implementation.template_arguments["WARP_TIME_SLICING"] == int(
        time_slicing
    )
    assert [item.name for item in call.argument_classifications] == [
        *expected_runtime,
        "mode",
        "warp_time_slicing",
    ]
    assert [item.kind for item in call.argument_classifications] == [
        *([ArgumentKind.RUNTIME] * len(expected_runtime)),
        ArgumentKind.STATIC,
        ArgumentKind.STATIC,
    ]
    assert call.argument_classifications[-1].role is ParameterRole.CONSTANT


def test_group_exchange_preserves_input_and_cache_relevant_policy():
    with pytest.raises(ValueError, match="out-of-place"):
        _exchange(value_form="in_place")

    first = _plan(this_block(), _exchange(), 64)
    equivalent = _plan(this_block(), _exchange(), 64)
    different_items = _plan(this_block(), _exchange(items_per_thread=4), 64)
    different_dtype = _plan(this_block(), _exchange(dtype="i64"), 64)
    sliced = _plan(this_block(), _exchange(warp_time_slicing=True), 64)

    assert first.semantic_key == equivalent.semantic_key
    assert first.artifact_key == equivalent.artifact_key
    assert first.artifact_key != different_items.artifact_key
    assert first.artifact_key != different_dtype.artifact_key
    assert first.artifact_key != sliced.artifact_key
    assert first.implementation.parameters[0][1].name == "input_items"
    assert not first.implementation.parameters[0][1].is_inout
    assert first.implementation.parameters[0][2].is_output


def test_group_exchange_rejects_invalid_group_and_warp_variants():
    logical = this_warp().group_by(8)
    block_only = _plan(logical, _exchange("warp_striped_to_blocked"), 64)
    time_sliced = _plan(
        logical,
        _exchange(warp_time_slicing=True),
        64,
    )
    grid = _plan(this_grid(), _exchange(), 64)
    misaligned = _plan(this_block(), _exchange("warp_striped_to_blocked"), 48)

    assert block_only.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert time_sliced.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert grid.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert misaligned.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT

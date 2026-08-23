# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group exchange planner contracts."""

import pytest

from tests.support.group_planning import (
    GroupExchangeMode,
    GroupLoweringTarget,
    GroupOperandKind,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    _exchange,
    _plan,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_warp,
)


@pytest.mark.parametrize(
    ("group", "target", "struct_name", "template_arguments", "barrier"),
    [
        (
            this_block(),
            GroupLoweringTarget.CUB_BLOCK,
            "BlockExchange",
            {
                "T": "int",
                "BLOCK_DIM_X": 128,
                "ITEMS_PER_THREAD": 3,
                "WARP_TIME_SLICING": 0,
                "BLOCK_DIM_Y": 1,
                "BLOCK_DIM_Z": 1,
            },
            SynchronizationScope.BLOCK,
        ),
        (
            this_warp(),
            GroupLoweringTarget.CUB_WARP,
            "WarpExchange",
            {
                "T": "int",
                "ITEMS_PER_THREAD": 3,
                "LOGICAL_WARP_THREADS": 32,
                "WARP_EXCHANGE_ALGORITHM": "::cub::WARP_EXCHANGE_SMEM",
            },
            SynchronizationScope.WARP,
        ),
    ],
)
@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        (GroupExchangeMode.STRIPED_TO_BLOCKED, "StripedToBlocked"),
        (GroupExchangeMode.BLOCKED_TO_STRIPED, "BlockedToStriped"),
    ],
)
def test_exchange_selects_exact_array_cub_with_implementation_storage(
    group,
    target,
    struct_name,
    template_arguments,
    barrier,
    mode,
    method_name,
):
    operation = _exchange(mode.value, 3)
    plan = _plan(group, operation, 128)

    assert plan.target is target
    assert plan.implementation.struct_name == struct_name
    assert plan.implementation.method_name == method_name
    assert plan.implementation.template_arguments == template_arguments
    assert plan.result.operand_kind is GroupOperandKind.ARRAY
    assert plan.result.result_items_per_thread == 3
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space is None
    assert plan.temp_storage.cpp_type is None
    assert plan.temp_storage.instances is None
    assert plan.temp_storage.instance_index is None
    assert not plan.temp_storage.exact_layout_required
    assert plan.synchronization.storage_reuse_barrier is barrier


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_group_exchange_leaves_frontend_item_qualification_out_of_core(group):
    plan = _plan(group, _exchange(items_per_thread=6), 64)

    assert plan.target in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }
    assert plan.result.result_items_per_thread == 6


@pytest.mark.parametrize(
    ("mode", "method_name", "rank_dtype", "valid_flag_dtype", "time_slicing"),
    [
        ("warp_striped_to_blocked", "WarpStripedToBlocked", None, None, True),
        ("blocked_to_warp_striped", "BlockedToWarpStriped", None, None, True),
        ("scatter_to_blocked", "ScatterToBlocked", "int", None, True),
        ("scatter_to_striped", "ScatterToStriped", "int", None, True),
        (
            "scatter_to_striped_guarded",
            "ScatterToStripedGuarded",
            "int",
            None,
            False,
        ),
        (
            "scatter_to_striped_flagged",
            "ScatterToStripedFlagged",
            "int",
            "unsigned char",
            False,
        ),
    ],
)
def test_group_exchange_plans_every_qualified_block_mode(
    mode,
    method_name,
    rank_dtype,
    valid_flag_dtype,
    time_slicing,
):
    operation = _exchange(
        mode,
        2,
        rank_dtype=rank_dtype,
        valid_flag_dtype=valid_flag_dtype,
        warp_time_slicing=time_slicing,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.method_name == method_name
    assert plan.implementation.template_arguments["WARP_TIME_SLICING"] == int(
        time_slicing
    )
    assert [item.name for item in call.argument_classifications] == [
        "value",
        "mode",
        *(("ranks",) if rank_dtype is not None else ()),
        *(("valid_flags",) if valid_flag_dtype is not None else ()),
    ]


def test_logical_warp_exchange_uses_mapped_width_and_rejects_block_only_modes():
    group = this_warp().group_by(8)
    plan = _plan(
        group,
        _exchange("scatter_to_striped", 2, rank_dtype="int"),
        64,
    )
    block_only = _plan(group, _exchange("warp_striped_to_blocked", 2), 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.method_name == "ScatterToStriped"
    assert plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == 8
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.instances is None
    assert block_only.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT

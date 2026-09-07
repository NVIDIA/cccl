# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group adjacent difference planner contracts."""

from tests.support.group_planning import (
    BlockAdjacentDifferenceDirection,
    GroupLoweringTarget,
    GroupOperandKind,
    LaunchFacts,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    _adjacent_difference,
    _plan,
    this_block,
    this_warp,
)


def test_adjacent_difference_selects_exact_multidimensional_block_cub():
    operation = _adjacent_difference(
        dtype="int",
        items_per_thread=3,
        direction=BlockAdjacentDifferenceDirection.LEFT,
        valid_items=17,
        tile_predecessor_item=0,
    )
    plan = _plan(this_block(), operation, (8, 4, 2))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockAdjacentDifference"
    assert plan.implementation.method_name == "SubtractLeftPartialTile"
    assert plan.implementation.template_arguments == {
        "T": "int",
        "BLOCK_DIM_X": 8,
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
        "ITEMS_PER_THREAD": 3,
    }
    assert plan.provenance.library == "CUB"
    assert plan.provenance.header == "cub/block/block_adjacent_difference.cuh"
    assert plan.provenance.cpp_class == "cub::BlockAdjacentDifference"
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.operand_kind is GroupOperandKind.ARRAY
    assert plan.result.result_items_per_thread == 3
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "tile_predecessor_item",
    )


def test_adjacent_difference_runtime_payloads_do_not_fragment_artifacts():
    first = _plan(
        this_block(),
        _adjacent_difference(valid_items=17, tile_predecessor_item=1),
        (64, 1, 1),
    )
    second = _plan(
        this_block(),
        _adjacent_difference(valid_items=31, tile_predecessor_item=999),
        (64, 1, 1),
    )

    assert first.semantic_key == second.semantic_key
    assert first.artifact_key == second.artifact_key


def test_adjacent_difference_rejects_non_block_groups_and_missing_exact_shape():
    warp = _plan(this_warp(), _adjacent_difference(), 64)
    missing = _plan(
        this_block(),
        _adjacent_difference(),
        LaunchFacts(max_block_dim=64),
    )

    assert warp.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM

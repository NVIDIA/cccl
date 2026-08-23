# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group radix planner contracts."""

from tests.support.group_planning import (
    GroupLoweringTarget,
    LaunchFacts,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    _plan,
    _radix_rank,
    _radix_sort,
    this_block,
    this_warp,
)


def test_radix_sort_selects_one_multidimensional_cub_artifact_and_two_results():
    plan = _plan(
        this_block(),
        _radix_sort(value_dtype="double", items_per_thread=3, descending=True),
        (8, 4, 2),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockRadixSort"
    assert plan.implementation.method_name == "SortDescending"
    assert plan.implementation.template_arguments["BLOCK_DIM_X"] == 8
    assert plan.implementation.template_arguments["BLOCK_DIM_Y"] == 4
    assert plan.implementation.template_arguments["BLOCK_DIM_Z"] == 2
    assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 3
    assert plan.provenance.header == "cub/block/block_radix_sort.cuh"
    assert plan.provenance.cpp_class == "cub::BlockRadixSort"
    assert [result.name for result in plan.result.values] == ["keys", "values"]
    assert all(
        result.visibility is ResultVisibility.PER_MEMBER
        for result in plan.result.values
    )
    assert plan.participation.uniform_arguments == ("begin_bit", "end_bit")
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_radix_sort_runtime_bit_values_do_not_enter_artifact_identity():
    operation = _radix_sort(items_per_thread=2)
    first = _plan(this_block(), operation, (64, 1, 1))
    second = _plan(this_block(), operation, (64, 1, 1))

    assert first.artifact_key == second.artifact_key
    assert first.call.argument_classifications[1].kind.name == "RUNTIME"
    assert operation.primitive.bit_range.begin_bit.value is None
    assert operation.primitive.bit_range.end_bit.value is None


def test_radix_rank_static_width_owns_prefix_result_and_public_cub_plan():
    plan = _plan(
        this_block(),
        _radix_rank(prefix_items=4, items_per_thread=2),
        (64, 1, 1),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockRadixRank"
    assert plan.implementation.method_name == "RankKeys"
    assert plan.implementation.template_arguments["RADIX_BITS"] == 8
    assert plan.implementation.template_arguments["BLOCK_DIM_X"] == 64
    assert plan.provenance.header == "cub/block/block_radix_rank.cuh"
    assert plan.provenance.cpp_class == "cub::BlockRadixRank"
    assert [result.name for result in plan.result.values] == [
        "ranks",
        "exclusive_digit_prefix",
    ]
    assert plan.result.values[1].items_per_member == 4

    wide_plan = _plan(
        this_block(),
        _radix_rank(end_bit=12, prefix_items=64, items_per_thread=2),
        (64, 1, 1),
    )
    assert wide_plan.target is GroupLoweringTarget.CUB_BLOCK
    assert wide_plan.implementation.template_arguments["RADIX_BITS"] == 12
    assert wide_plan.result.values[1].items_per_member == 64


def test_radix_plans_reject_wrong_group_missing_exact_shape_and_oversized_tile():
    wrong_group = _plan(this_warp(), _radix_sort(), 64)
    missing = _plan(
        this_block(),
        _radix_rank(block_threads=64),
        LaunchFacts(max_block_dim=64),
    )
    oversized = _plan(
        this_block(),
        _radix_sort(items_per_thread=2),
        (32768, 1, 1),
    )

    assert wrong_group.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    assert oversized.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "<= 65535" in oversized.unsupported.message

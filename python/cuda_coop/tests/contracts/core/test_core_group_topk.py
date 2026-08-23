# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group TopK planner contracts."""

from tests.support.group_planning import (
    GroupLoweringTarget,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    _plan,
    _topk,
    this_block,
    this_warp,
)


def test_topk_preserves_block_spec_identity_and_pair_result_contract():
    operation = _topk(value_dtype="float", items_per_thread=3, selection="min")
    plan = _plan(this_block(), operation, (64, 1, 1))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert operation.semantic_key == operation.primitive.semantic_key
    assert plan.implementation.semantic_key == operation.primitive.semantic_key
    assert plan.semantic_key[1] == operation.primitive.semantic_key
    assert plan.artifact_key[3] == operation.primitive.semantic_key
    assert plan.artifact_key[4] == operation.primitive.semantic_key
    assert plan.implementation.method_name == "min_pairs_partial"
    assert plan.provenance.cpp_class == "cub::BlockTopKCoop"
    assert [result.name for result in plan.result.values] == ["keys", "values"]
    assert all(
        result.visibility is ResultVisibility.PER_MEMBER
        for result in plan.result.values
    )
    assert all(result.items_per_member == 3 for result in plan.result.values)
    assert plan.participation.uniform_arguments == (
        "k",
        "num_valid",
        "begin_bit",
        "end_bit",
    )
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_topk_rejects_wrong_group_and_specialization_launch_mismatch():
    wrong_group = _plan(this_warp(), _topk(), (64, 1, 1))
    wrong_launch = _plan(
        this_block(),
        _topk(block_dim=(32, 1, 1)),
        (64, 1, 1),
    )

    assert wrong_group.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert wrong_launch.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "must match" in wrong_launch.unsupported.message

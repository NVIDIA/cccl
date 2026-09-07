# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group merge sort planner contracts."""

import pytest

from tests.support.group_planning import (
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    _merge_sort,
    _plan,
    this_block,
    this_thread,
    this_warp,
)


def test_block_merge_sort_plans_one_multidimensional_public_cub_artifact():
    plan = _plan(
        this_block(),
        _merge_sort(
            key_dtype="int",
            value_dtype="float",
            items_per_thread=3,
            descending=True,
            valid_items=17,
            oob_default=-1,
        ),
        (8, 4, 2),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockMergeSort"
    assert plan.implementation.method_name == "Sort"
    assert plan.implementation.template_arguments == {
        "KeyT": "int",
        "BLOCK_DIM_X": 8,
        "ITEMS_PER_THREAD": 3,
        "ValueT": "float",
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
    }
    assert plan.provenance.header == "cub/block/block_merge_sort.cuh"
    assert plan.provenance.cpp_class == "cub::BlockMergeSort"
    assert [result.name for result in plan.result.values] == ["keys", "values"]
    assert [result.dtype for result in plan.result.values] == ["int", "float"]
    assert all(result.items_per_member == 3 for result in plan.result.values)
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
    )
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_warp_merge_sort_plans_physical_and_logical_storage_partitions():
    physical = _plan(this_warp(), _merge_sort(items_per_thread=2), 64)
    logical = _plan(
        this_warp().group_by(16),
        _merge_sort(value_dtype="float", items_per_thread=2),
        64,
    )
    physical_partial = _plan(
        this_warp(),
        _merge_sort(
            value_dtype="float",
            items_per_thread=2,
            valid_items=59,
            oob_default=999,
        ),
        64,
    )

    assert physical.target is GroupLoweringTarget.CUB_WARP
    assert physical.implementation.struct_name == "WarpMergeSort"
    assert physical.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 32
    assert physical.provenance.header == "cub/warp/warp_merge_sort.cuh"
    assert physical.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
    assert logical.target is GroupLoweringTarget.CUB_WARP
    assert logical.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 16
    assert [result.name for result in logical.result.values] == ["keys", "values"]
    assert physical_partial.target is GroupLoweringTarget.CUB_WARP
    assert [
        parameter.name for parameter in physical_partial.implementation.parameters[0]
    ] == [
        "temp_storage",
        "keys",
        "values",
        "compare_op",
        "valid_items",
        "oob_default",
    ]
    assert physical_partial.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
    )
    precondition = physical_partial.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (0, 64)
    precondition.validate(0)
    precondition.validate(64)
    for invalid in (-1, 65):
        with pytest.raises(ValueError, match="valid_items must be"):
            precondition.validate(invalid)


def test_merge_sort_runtime_values_do_not_fragment_artifacts():
    first = _plan(
        this_block(),
        _merge_sort(valid_items=17, oob_default=-1),
        64,
    )
    second = _plan(
        this_block(),
        _merge_sort(valid_items=31, oob_default=999),
        64,
    )

    assert first.artifact_key == second.artifact_key

    warp_first = _plan(
        this_warp(),
        _merge_sort(valid_items=17, oob_default=-1),
        64,
    )
    warp_second = _plan(
        this_warp(),
        _merge_sort(valid_items=31, oob_default=999),
        64,
    )
    assert warp_first.artifact_key == warp_second.artifact_key


def test_merge_sort_rejects_non_power_of_two_blocks_and_incomplete_warps():
    wrong_group = _plan(this_thread(), _merge_sort(), 1)
    missing_exact = _plan(
        this_block(),
        _merge_sort(),
        LaunchFacts(max_block_dim=64),
    )
    block = _plan(this_block(), _merge_sort(), 48)
    physical_warp = _plan(this_warp(), _merge_sort(), 48)
    logical_warp = _plan(this_warp().group_by(16), _merge_sort(), 24)

    assert wrong_group.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert (
        missing_exact.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    )
    assert block.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert physical_warp.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
    assert logical_warp.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group discontinuity planner contracts."""

import pytest

from tests.support.group_planning import (
    BlockDiscontinuityMode,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    _discontinuity,
    _plan,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
)


@pytest.mark.parametrize(
    ("mode", "predecessor", "successor", "result_names"),
    [
        (BlockDiscontinuityMode.HEADS, None, None, ("head_flags",)),
        (BlockDiscontinuityMode.HEADS, 1, None, ("head_flags",)),
        (BlockDiscontinuityMode.TAILS, None, None, ("tail_flags",)),
        (BlockDiscontinuityMode.TAILS, None, 9, ("tail_flags",)),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            None,
            None,
            ("head_flags", "tail_flags"),
        ),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            1,
            None,
            ("head_flags", "tail_flags"),
        ),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            None,
            9,
            ("head_flags", "tail_flags"),
        ),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            1,
            9,
            ("head_flags", "tail_flags"),
        ),
    ],
)
def test_discontinuity_plans_every_public_cub_boundary_overload(
    mode,
    predecessor,
    successor,
    result_names,
):
    operation = _discontinuity(
        items_per_thread=3,
        mode=mode,
        tile_predecessor_item=predecessor,
        tile_successor_item=successor,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=(8, 4, 2)))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockDiscontinuity"
    assert plan.implementation.method_name == mode.cub_method_name
    assert plan.implementation.template_arguments["BLOCK_DIM_X"] == 8
    assert plan.implementation.template_arguments["BLOCK_DIM_Y"] == 4
    assert plan.implementation.template_arguments["BLOCK_DIM_Z"] == 2
    assert plan.provenance.header == "cub/block/block_discontinuity.cuh"
    assert tuple(result.name for result in plan.result.values) == result_names
    assert all(
        result.dtype == "flag" and result.items_per_member == 3
        for result in plan.result.values
    )
    expected_uniform = (
        *(("tile_predecessor_item",) if predecessor is not None else ()),
        *(("tile_successor_item",) if successor is not None else ()),
    )
    assert plan.participation.uniform_arguments == expected_uniform
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_discontinuity_boundary_values_do_not_fragment_artifacts():
    first = _plan(
        this_block(),
        _discontinuity(
            mode="heads_and_tails",
            tile_predecessor_item=1,
            tile_successor_item=9,
        ),
        64,
    )
    second = _plan(
        this_block(),
        _discontinuity(
            mode="heads_and_tails",
            tile_predecessor_item=111,
            tile_successor_item=999,
        ),
        64,
    )

    assert first.artifact_key == second.artifact_key

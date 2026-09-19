# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Neighbor topology, boundary, output, and partial-count contracts."""

import pytest

from cuda import coop
from cuda.coop._core import (
    FLOAT64,
    INT32,
    ArgumentBinding,
    CxxOperator,
    Dependency,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block.neighbors import (
    BlockNeighborSemantics,
    make_block_neighbor_spec,
)
from cuda.coop._core.group.neighbors import GroupNeighborSemantics


def _primitive(operation="adjacent_difference", mode="left", **kwargs):
    return BlockNeighborSemantics(
        operation=operation,
        dtype=FLOAT64,
        items_per_thread=3,
        mode=mode,
        operator=CxxOperator("::cuda::std::minus<T>", Dependency("T"), "op"),
        **kwargs,
    )


@pytest.mark.parametrize(
    "operation,mode,names",
    [
        ("adjacent_difference", "left", ("differences",)),
        ("adjacent_difference", "right", ("differences",)),
        ("discontinuity", "heads", ("heads",)),
        ("discontinuity", "tails", ("tails",)),
        ("discontinuity", "heads_and_tails", ("heads", "tails")),
    ],
)
def test_result_contract_and_non_power_of_two_block(operation, mode, names):
    primitive = _primitive(operation, mode)
    plan = plan_group_primitive(
        make_group_primitive_call(coop.this_block(), GroupNeighborSemantics(primitive)),
        LaunchFacts(exact_block_dim=(5, 3, 2)),
    ).require_supported()
    assert primitive.result_names == names
    assert primitive.result_dtype == (
        FLOAT64 if operation == "adjacent_difference" else INT32
    )
    assert len(plan.result.values) == len(names)
    assert plan.participation.exact_block_dim == (5, 3, 2)
    assert all(output.items_per_member == 3 for output in plan.result.values)


@pytest.mark.parametrize(
    "group", [coop.this_warp(), coop.this_warp().group_by(8), coop.this_grid()]
)
def test_reject_other_groups(group):
    plan = plan_group_primitive(
        make_group_primitive_call(group, GroupNeighborSemantics(_primitive())),
        LaunchFacts(exact_block_dim=(64, 1, 1)),
    )
    assert plan.unsupported is not None


@pytest.mark.parametrize("count", [0, 1, 89, 90])
def test_partial_count_and_uniform_boundaries(count):
    call = GroupNeighborSemantics(
        _primitive(partial=True, predecessor=True),
        valid_items=ArgumentBinding.static(count),
    )
    plan = plan_group_primitive(
        make_group_primitive_call(coop.this_block(), call),
        LaunchFacts(exact_block_dim=(5, 3, 2)),
    ).require_supported()
    assert set(plan.participation.uniform_arguments) == {
        "valid_items",
        "tile_predecessor_item",
    }


@pytest.mark.parametrize("count", [-1, 91, 2**32])
def test_reject_invalid_static_count(count):
    with pytest.raises(ValueError, match="valid_items"):
        call = GroupNeighborSemantics(
            _primitive(partial=True), valid_items=ArgumentBinding.static(count)
        )
        plan_group_primitive(
            make_group_primitive_call(coop.this_block(), call),
            LaunchFacts(exact_block_dim=(30, 1, 1)),
        )


@pytest.mark.parametrize(
    "operation,mode,options",
    [
        ("adjacent_difference", "left", {"successor": True}),
        ("adjacent_difference", "right", {"predecessor": True}),
        ("adjacent_difference", "right", {"partial": True, "successor": True}),
        ("discontinuity", "heads", {"successor": True}),
        ("discontinuity", "tails", {"predecessor": True}),
        ("discontinuity", "heads", {"partial": True}),
    ],
)
def test_unsupported_boundary_combinations(operation, mode, options):
    with pytest.raises(ValueError):
        _primitive(operation, mode, **options)


def test_boundary_presence_and_partial_policy_change_identity():
    calls = [_primitive(), _primitive(predecessor=True), _primitive(partial=True)]
    assert len({call.semantic_key for call in calls}) == 3
    specs = [make_block_neighbor_spec(call, block_dim=(32, 1, 1)) for call in calls]
    assert len({spec.semantic_key for spec in specs}) == 3


def test_common_rejects_backend_callback_and_partial_discontinuity():
    with pytest.raises(TypeError, match="difference_op"):
        coop.adjacent_difference(
            coop.this_block(), object(), difference_op=lambda a, b: a - b
        )
    with pytest.raises(TypeError, match="valid_items"):
        coop.discontinuity(coop.this_block(), object(), valid_items=3)

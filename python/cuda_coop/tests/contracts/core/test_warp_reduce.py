# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from cuda.coop._core import (
    ArgumentBinding,
    BlockReduceOperator,
    GroupLoweringTarget,
    GroupReduceSemantics,
    LaunchFactOrigin,
    LaunchFacts,
    SynchronizationScope,
    WarpReduceOperation,
    make_group_primitive_call,
    make_warp_reduce_spec,
    plan_group_primitive,
    this_warp,
)
from cuda.coop._core.group import UnsupportedReasonCode


def _launch(block_dim):
    return LaunchFacts(
        exact_block_dim=block_dim,
        provenance=LaunchFactOrigin(
            fact="exact_block_dim",
            source="test",
            verified=True,
        ),
    )


def _plan(operation: GroupReduceSemantics, block_dim=(8, 4, 2)):
    return plan_group_primitive(
        make_group_primitive_call(this_warp(), operation),
        _launch(block_dim),
    )


@pytest.mark.parametrize("block_dim", (32, 64, (8, 8), (8, 4, 2), (4, 4, 4)))
def test_physical_warp_sum_accepts_complete_multidimensional_blocks(
    block_dim,
) -> None:
    plan = _plan(
        GroupReduceSemantics(dtype="int32", operation="sum"),
        block_dim=block_dim,
    ).require_supported()

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.resolved_group.kind == "warp"
    assert plan.resolved_group.static_size == 32
    assert plan.participation is not None
    assert plan.participation.exact_group_size == 32
    assert plan.participation.complete_membership
    assert plan.synchronization is not None
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
    assert plan.implementation is not None
    assert plan.implementation.operation is WarpReduceOperation.SUM
    assert plan.implementation.method_name == "Sum"
    expected_warps = 1 if block_dim == 32 else 2
    assert plan.implementation.warp_count == expected_warps
    assert plan.provenance is not None
    assert plan.provenance.header == "cub/warp/warp_reduce.cuh"
    assert plan.provenance.cpp_class == "cub::WarpReduce"
    assert plan.provenance.method == "Sum"


def test_reduce_with_sum_operator_selects_cub_sum_name() -> None:
    plan = _plan(
        GroupReduceSemantics(dtype="float32", operation="reduce", binary_op="sum")
    ).require_supported()

    assert plan.implementation is not None
    assert plan.implementation.operation is WarpReduceOperation.REDUCE
    assert plan.implementation.binary_op is BlockReduceOperator.SUM
    assert plan.implementation.method_name == "Sum"
    assert plan.provenance is not None
    assert plan.provenance.method == "Sum"


def test_max_operator_selects_cub_reduce() -> None:
    plan = _plan(
        GroupReduceSemantics(dtype="int32", binary_op="max")
    ).require_supported()

    assert plan.implementation is not None
    assert plan.implementation.binary_op is BlockReduceOperator.MAX
    assert plan.implementation.method_name == "Reduce"


@pytest.mark.parametrize("valid_items", (1, 17, 32))
def test_static_valid_prefix_is_checked_against_physical_warp(
    valid_items,
) -> None:
    plan = _plan(
        GroupReduceSemantics(
            dtype="uint32",
            operation="sum",
            valid_items=ArgumentBinding.static(valid_items),
        )
    ).require_supported()

    assert plan.participation is not None
    assert plan.participation.uniform_arguments == ("valid_items",)
    assert plan.participation.valid_member_selection == ("first valid_items warp lanes")


def test_static_valid_prefix_cannot_exceed_physical_warp() -> None:
    with pytest.raises(ValueError, match="valid_items must be at most 32"):
        _plan(
            GroupReduceSemantics(
                dtype="uint32",
                operation="sum",
                valid_items=ArgumentBinding.static(33),
            )
        )


def test_explicit_block_algorithm_is_rejected_for_warp() -> None:
    with pytest.raises(ValueError, match="applies to block groups"):
        _plan(
            GroupReduceSemantics(
                dtype="int32",
                operation="sum",
                algorithm="warp_reductions",
            )
        )


@pytest.mark.parametrize("block_dim", (16, 31, 33, 48, (3, 4, 4)))
def test_partial_physical_warp_launch_is_typed_unsupported(block_dim) -> None:
    plan = _plan(
        GroupReduceSemantics(dtype="int32", operation="sum"),
        block_dim=block_dim,
    )

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported is not None
    assert plan.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
    with pytest.raises(NotImplementedError, match="complete 32-thread warps"):
        plan.require_supported()


def test_warp_spec_identity_tracks_enclosing_block_shape() -> None:
    first = make_warp_reduce_spec(dtype="int32", block_dim=(8, 8))
    second = make_warp_reduce_spec(dtype="int32", block_dim=(4, 4, 4))

    assert first.semantic_key != second.semantic_key
    assert first.warp_count == second.warp_count == 2


def test_warp_spec_rejects_partial_enclosing_block() -> None:
    with pytest.raises(ValueError, match="complete 32-thread physical warps"):
        make_warp_reduce_spec(dtype="int32", block_dim=48)

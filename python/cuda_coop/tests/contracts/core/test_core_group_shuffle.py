# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group shuffle planner contracts."""

import pytest

from tests.support.group_planning import (
    ArgumentBinding,
    ArgumentKind,
    GroupLoweringTarget,
    GroupOperandKind,
    LaunchFacts,
    SynchronizationScope,
    UnsupportedReasonCode,
    _plan,
    _shuffle,
    this_block,
)


@pytest.mark.parametrize(
    ("operation", "method", "result_kind", "result_names"),
    [
        (_shuffle(mode="offset"), "Offset", GroupOperandKind.SCALAR, ("value",)),
        (_shuffle(mode="rotate"), "Rotate", GroupOperandKind.SCALAR, ("value",)),
        (
            _shuffle(items_per_thread=3, mode="up"),
            "Up",
            GroupOperandKind.ARRAY,
            ("value",),
        ),
        (
            _shuffle(items_per_thread=3, mode="up", block_suffix=True),
            "Up",
            GroupOperandKind.ARRAY,
            ("value", "block_suffix"),
        ),
        (
            _shuffle(items_per_thread=3, mode="down", block_prefix=True),
            "Down",
            GroupOperandKind.ARRAY,
            ("value", "block_prefix"),
        ),
    ],
)
def test_shuffle_plans_only_public_cub_shapes(
    operation,
    method,
    result_kind,
    result_names,
):
    plan = _plan(this_block(), operation, (8, 4, 2))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockShuffle"
    assert plan.implementation.method_name == method
    assert plan.implementation.template_arguments["BLOCK_DIM_Y"] == 4
    assert plan.implementation.template_arguments["BLOCK_DIM_Z"] == 2
    assert plan.provenance.header == "cub/block/block_shuffle.cuh"
    assert plan.result.primary.operand_kind is result_kind
    assert tuple(result.name for result in plan.result.values) == result_names
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_shuffle_scalar_distance_is_runtime_and_outside_artifact_identity():
    first = _plan(this_block(), _shuffle(mode="rotate"), 64)
    second = _plan(this_block(), _shuffle(mode="rotate"), 64)

    assert first.participation.uniform_arguments == ("distance",)
    assert first.artifact_key == second.artifact_key
    assert first.call.argument_classifications[-1].name == "distance"
    assert first.call.argument_classifications[-1].kind is ArgumentKind.RUNTIME


def test_shuffle_planner_rejects_non_cub_shapes_and_missing_exact_shape():
    scalar_up = _plan(this_block(), _shuffle(mode="up"), 64)
    array_rotate = _plan(
        this_block(),
        _shuffle(
            items_per_thread=2,
            mode="rotate",
            distance=ArgumentBinding.omitted(),
        ),
        64,
    )
    missing = _plan(
        this_block(),
        _shuffle(mode="offset"),
        LaunchFacts(max_block_dim=64),
    )

    assert scalar_up.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert array_rotate.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM

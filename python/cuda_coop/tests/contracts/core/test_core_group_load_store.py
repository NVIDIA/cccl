# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group load store planner contracts."""

import pytest

from tests.support.group_planning import (
    ArgumentBinding,
    GroupLoadStoreAlgorithm,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    UnsupportedReasonCode,
    _load_store,
    _plan,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_warp,
)


@pytest.mark.parametrize(
    ("group", "kind", "target", "cpp_class"),
    [
        (this_block(), "load", GroupLoweringTarget.CUB_BLOCK, "cub::BlockLoad"),
        (this_block(), "store", GroupLoweringTarget.CUB_BLOCK, "cub::BlockStore"),
        (this_warp(), "load", GroupLoweringTarget.CUB_WARP, "cub::WarpLoad"),
        (this_warp(), "store", GroupLoweringTarget.CUB_WARP, "cub::WarpStore"),
    ],
)
def test_group_load_store_selects_real_cub(group, kind, target, cpp_class):
    plan = _plan(group, _load_store(kind, items_per_thread=3), 64)

    assert plan.target is target
    assert plan.provenance.library == "CUB"
    assert plan.provenance.cpp_class == cpp_class
    if kind == "load":
        assert plan.result.result_items_per_thread == 3
    else:
        assert plan.result is None


def test_group_load_models_partial_tile_and_offset_bindings():
    operation = _load_store(
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(0),
        offset=ArgumentBinding.static(4),
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert (
        plan.participation.valid_member_selection == "first valid_items tile elements"
    )
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
        "offset",
    )
    assert [
        classification.name for classification in call.argument_classifications
    ] == [
        "source",
        "valid_items",
        "oob_default",
        "offset",
        "algorithm",
    ]


def test_group_load_store_supports_logical_warps_and_rejects_invalid_algorithms():
    mapped = this_warp().group_by(8)
    mapped_plan = _plan(mapped, _load_store(), 64)
    warp_plan = _plan(
        this_warp(),
        _load_store(algorithm=GroupLoadStoreAlgorithm.WARP_TRANSPOSE),
        64,
    )

    assert mapped_plan.target is GroupLoweringTarget.CUB_WARP
    assert mapped_plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == 8
    assert mapped_plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert mapped_plan.temp_storage.instances is None
    assert warp_plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT

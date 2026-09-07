# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group histogram planner contracts."""

from tests.support.group_planning import (
    GroupLoweringTarget,
    UnsupportedReasonCode,
    _histogram,
    _plan,
    this_block,
    this_warp,
)


def test_histogram_planner_records_capacity_and_sample_preconditions():
    plan = _plan(this_block(), _histogram(), 64)

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.result.primary.items_per_member == 1
    assert plan.result.primary.dtype == "unsigned int"
    assert [
        (condition.name, condition.minimum, condition.maximum)
        for condition in plan.participation.argument_preconditions
    ] == [("samples", 0, 63)]

    unsupported = _plan(this_warp(), _histogram(), 64)
    assert unsupported.target is GroupLoweringTarget.UNSUPPORTED
    assert unsupported.unsupported.code is UnsupportedReasonCode.GROUP_KIND

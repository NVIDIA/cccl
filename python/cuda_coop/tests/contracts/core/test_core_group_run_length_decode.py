# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group run length decode planner contracts."""

from tests.support.group_planning import (
    GroupLoweringTarget,
    UnsupportedReasonCode,
    _plan,
    _run_length_decode,
    this_block,
    this_warp,
)


def test_run_length_decode_planner_records_all_logical_results():
    plan = _plan(this_block(), _run_length_decode(), 64)

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert [result.name for result in plan.result.values] == [
        "decoded_items",
        "relative_offsets",
        "total_decoded_size",
    ]
    assert plan.participation.uniform_arguments == ("decoded_window_offset",)
    assert [
        condition.name for condition in plan.participation.argument_preconditions
    ] == [
        "run_lengths",
        "sum(run_lengths)",
        "decoded_window_offset",
    ]

    unsupported = _plan(this_warp(), _run_length_decode(), 64)
    assert unsupported.target is GroupLoweringTarget.UNSUPPORTED
    assert unsupported.unsupported.code is UnsupportedReasonCode.GROUP_KIND

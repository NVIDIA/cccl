# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core import (
    LaunchFacts,
    UnsupportedReasonCode,
    merge_launch_facts,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
)
from tests._group_planning import _load_store, _plan


def test_maximum_block_dimensions_do_not_substitute_for_exact_dimensions():
    maximum = LaunchFacts(max_block_dim=(16, 8, 2))
    unsupported = _plan(this_block(), _load_store(), maximum)
    assert (
        unsupported.unsupported.code
        is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    )

    merged = merge_launch_facts(maximum, LaunchFacts(exact_block_dim=(8, 4, 2)))
    supported = _plan(this_block(), _load_store(), merged)
    arguments = supported.implementation.template_arguments
    assert tuple(arguments[f"BLOCK_DIM_{axis}"] for axis in "XYZ") == (8, 4, 2)


@pytest.mark.parametrize("group", [this_thread(), this_cluster(), this_grid()])
def test_non_load_store_targets_are_typed_unsupported_before_resolution(group):
    plan = _plan(group, _load_store(), LaunchFacts(exact_block_dim=48))

    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert plan.artifact_key is None

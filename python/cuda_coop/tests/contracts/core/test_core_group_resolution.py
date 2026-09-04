# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launch and group resolution contracts used by Block Load/Store."""

import pytest

from cuda.coop._core import (
    GroupLoweringTarget,
    LaunchFactOrigin,
    LaunchFacts,
    UnsupportedReasonCode,
    merge_launch_facts,
    resolve_thread_group,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
)
from tests.support.group_planning import _load_store, _plan


def test_launch_facts_keep_exact_bounds_and_provenance_distinct():
    exact = LaunchFacts(
        exact_block_dim=(8, 4),
        max_block_dim=(16, 8),
        provenance=LaunchFactOrigin("exact_block_dim", "call_metadata"),
    )
    same_facts = LaunchFacts(
        exact_block_dim=(8, 4),
        max_block_dim=(16, 8),
        provenance=LaunchFactOrigin("exact_block_dim", "reqntid"),
    )

    assert exact == same_facts
    assert exact.exact_block_dim == (8, 4, 1)
    assert exact.exact_block_threads == 32
    assert exact.max_block_threads == 128
    assert exact.provenance != same_facts.provenance


def test_launch_fact_merging_preserves_exact_dimensions():
    merged = merge_launch_facts(
        LaunchFacts(
            max_block_dim=(16, 8, 2),
            provenance=LaunchFactOrigin("max_block_dim", "kernel_attribute"),
        ),
        LaunchFacts(
            exact_block_dim=(8, 4, 2),
            provenance=LaunchFactOrigin("exact_block_dim", "launch_config"),
        ),
    )

    assert merged.exact_block_dim == (8, 4, 2)
    assert merged.max_block_dim == (16, 8, 2)


def test_block_load_store_requires_exact_not_maximum_dimensions():
    plan = _plan(
        this_block(),
        _load_store(),
        LaunchFacts(max_block_dim=(32, 2, 1)),
    )

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM


@pytest.mark.parametrize(
    "group",
    [this_thread(), this_cluster(), this_grid()],
)
def test_non_load_store_targets_are_typed_unsupported_before_resolution(group):
    plan = _plan(group, _load_store(), LaunchFacts(exact_block_dim=48))

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert plan.artifact_key is None
    assert "only this_block()" in plan.unsupported.message


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (64, (64, 1, 1)),
        ((8, 4), (8, 4, 1)),
        ((4, 4, 4), (4, 4, 4)),
    ],
)
def test_block_resolution_preserves_exact_launch_shape(shape, expected):
    resolved = resolve_thread_group(
        this_block(),
        LaunchFacts(exact_block_dim=shape),
    ).require_supported()

    assert resolved.hierarchy.block_dim == expected

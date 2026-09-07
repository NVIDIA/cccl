# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable group resolution planner contracts."""

import pytest

from tests.support.group_planning import (
    _COMPLETE_WARP_GROUP_SAMPLES,
    _NON_COMPLETE_WARP_GROUP_SAMPLES,
    COMPLETE_WARP_GROUP_KINDS,
    THREAD_GROUP_KINDS,
    GroupLoweringTarget,
    LaunchFactOrigin,
    LaunchFacts,
    UnsupportedReasonCode,
    _exchange,
    _plan,
    _reduce,
    merge_launch_facts,
    resolve_thread_group,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)


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
    assert hash(exact) == hash(same_facts)
    assert exact.exact_block_dim == (8, 4, 1)
    assert exact.exact_block_threads == 32
    assert exact.max_block_threads == 128
    assert exact.provenance != same_facts.provenance


def test_launch_fact_verification_is_diagnostic_not_semantic():
    asserted = LaunchFacts(
        cooperative_launch=True,
        provenance=LaunchFactOrigin(
            "cooperative_launch",
            "call_metadata",
        ),
    )
    verified = LaunchFacts(
        cooperative_launch=True,
        provenance=LaunchFactOrigin(
            "cooperative_launch",
            "kernel_launch_config",
            verified=True,
        ),
    )

    assert asserted == verified
    assert not asserted.is_verified("cooperative_launch")
    assert verified.is_verified("cooperative_launch")
    assert not verified.is_verified("cluster_launch")


def test_merge_launch_facts_reconciles_without_promoting_maximums():
    merged = merge_launch_facts(
        LaunchFacts(
            max_block_dim=(256, 8, 2),
            provenance=LaunchFactOrigin("max_block_dim", "maxntid:a"),
        ),
        LaunchFacts(
            max_block_dim=(128, 16, 1),
            cooperative_launch=True,
            provenance=LaunchFactOrigin("max_block_dim", "maxntid:b"),
        ),
    )

    assert merged.exact_block_dim is None
    assert merged.max_block_dim == (128, 8, 1)
    assert merged.cooperative_launch is True
    assert len(merged.provenance) == 2

    with pytest.raises(ValueError, match="conflicting exact_block_dim"):
        merge_launch_facts(
            LaunchFacts(exact_block_dim=32),
            LaunchFacts(exact_block_dim=64),
        )
    with pytest.raises(ValueError, match="conflicting cooperative_launch"):
        merge_launch_facts(
            LaunchFacts(cooperative_launch=True),
            LaunchFacts(cooperative_launch=False),
        )
    with pytest.raises(ValueError, match="exceeds max_block_dim"):
        LaunchFacts(exact_block_dim=(64, 2), max_block_dim=(32, 2))


def test_exact_launch_is_required_and_current_groups_are_resolved():
    operation = _reduce()
    missing = _plan(
        this_block(),
        operation,
        LaunchFacts(max_block_dim=256),
    )

    assert missing.target is GroupLoweringTarget.UNSUPPORTED
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    assert missing.artifact_key is None
    with pytest.raises(NotImplementedError, match="only an upper bound"):
        missing.require_supported()

    resolved = _plan(this_block(), operation, (8, 4, 1))
    assert resolved.resolved_group.block_dim == (8, 4, 1)
    assert resolved.resolved_group.source == "launch_facts"


def test_shared_group_resolution_builds_the_requested_enclosing_hierarchy():
    facts = LaunchFacts(
        exact_block_dim=(8, 4, 2),
        exact_grid_dim=8,
        cluster_launch=False,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )

    resolved = resolve_thread_group(
        this_thread(),
        facts,
        through_level="grid",
    ).require_supported()

    assert resolved.kind == "thread"
    assert resolved.hierarchy.block_dim == (8, 4, 2)
    assert resolved.hierarchy.cluster_dim == (1, 1, 1)
    assert resolved.hierarchy.grid_dim == (8, 1, 1)
    assert resolved.source == "launch_facts"


@pytest.mark.parametrize(
    ("group", "facts", "message"),
    (
        (
            this_cluster(),
            LaunchFacts(exact_block_dim=32, exact_cluster_dim=2),
            "backend-verified cluster launch state",
        ),
        (
            this_cluster(),
            LaunchFacts(
                exact_block_dim=32,
                exact_cluster_dim=2,
                cluster_launch=False,
                provenance=(
                    LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
                ),
            ),
            "multi-block cluster operations require verified cluster launch",
        ),
        (
            this_grid(),
            LaunchFacts(
                exact_block_dim=32,
                cluster_launch=False,
                provenance=(
                    LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
                ),
            ),
            "grid group operations require exact static grid dimensions",
        ),
        (
            this_grid(),
            LaunchFacts(
                exact_block_dim=32,
                exact_cluster_dim=2,
                exact_grid_dim=3,
                cluster_launch=True,
                provenance=(
                    LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
                ),
            ),
            "grid dimensions must be divisible by the cluster dimensions",
        ),
    ),
)
def test_shared_group_resolution_rejects_incomplete_launch_capabilities(
    group,
    facts,
    message,
):
    resolution = resolve_thread_group(group, facts)

    assert resolution.unsupported is not None
    assert resolution.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert message in resolution.unsupported.message


@pytest.mark.parametrize(
    "group",
    [
        this_thread(),
        this_block(),
        this_cluster(),
        this_grid(),
    ],
)
@pytest.mark.parametrize(("block_threads", "is_supported"), [(48, False), (64, True)])
def test_shared_group_resolution_requires_complete_warps_for_warp_queries(
    group,
    block_threads,
    is_supported,
):
    facts = LaunchFacts(
        exact_block_dim=block_threads,
        exact_grid_dim=8,
        cluster_launch=False,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )

    resolution = resolve_thread_group(group, facts, through_level="warp")

    if is_supported:
        assert resolution.require_supported().block_dim == (block_threads, 1, 1)
    else:
        assert (
            resolution.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
        )
        with pytest.raises(NotImplementedError, match="complete 32-thread warps"):
            resolution.require_supported()


def test_shared_group_samples_cover_complete_warp_partition():
    assert {
        group.kind for group in _COMPLETE_WARP_GROUP_SAMPLES
    } == COMPLETE_WARP_GROUP_KINDS
    assert {
        group.kind for group in _NON_COMPLETE_WARP_GROUP_SAMPLES
    } == THREAD_GROUP_KINDS - COMPLETE_WARP_GROUP_KINDS


@pytest.mark.parametrize(
    "group",
    _COMPLETE_WARP_GROUP_SAMPLES,
    ids=lambda group: group.kind,
)
def test_shared_group_resolution_enforces_every_complete_warp_group_kind(group):
    resolution = resolve_thread_group(
        group,
        LaunchFacts(exact_block_dim=48),
    )

    assert resolution.unsupported is not None
    assert resolution.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP


@pytest.mark.parametrize(
    "group",
    _NON_COMPLETE_WARP_GROUP_SAMPLES,
    ids=lambda group: group.kind,
)
def test_shared_group_resolution_allows_non_complete_warp_group_kinds(group):
    facts = LaunchFacts(
        exact_block_dim=48,
        exact_grid_dim=8,
        cluster_launch=False,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )

    resolved = resolve_thread_group(group, facts).require_supported()

    assert resolved.kind == group.kind
    if group.kind == "thread":
        assert resolved.is_current
    else:
        assert resolved.block_dim == (48, 1, 1)


def test_shared_group_resolution_reconciles_mapped_groups():
    group = this_block().group_by(2, exhaustive=False)

    resolved = resolve_thread_group(
        group,
        LaunchFacts(exact_block_dim=160),
    ).require_supported()

    assert resolved.kind == "warps_within_block"
    assert resolved.static_size == 64
    assert resolved.groups_per_parent == 2
    assert resolved.remainder_count == 1
    assert resolved.complete_membership is False


def test_partial_physical_warp_partition_fails_closed():
    plan = _plan(this_warp(), _reduce(), 48)
    mapped_plan = _plan(
        this_warp().group_by(8),
        _reduce(),
        48,
    )

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
    assert "complete 32-thread warps" in plan.unsupported.message
    assert mapped_plan.target is GroupLoweringTarget.UNSUPPORTED
    assert mapped_plan.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP


def test_exchange_requires_exact_launch_and_complete_physical_warps():
    missing_exact = _plan(
        this_block(),
        _exchange(),
        LaunchFacts(max_block_dim=128),
    )
    partial_warp = _plan(this_warp(), _exchange(), 48)

    assert (
        missing_exact.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    )
    assert "upper bound" in missing_exact.unsupported.message
    assert partial_warp.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
    assert "complete 32-thread warps" in partial_warp.unsupported.message


def test_cluster_resolution_rejects_unknown_launch_state():
    plan = _plan(
        this_cluster(),
        _reduce(),
        LaunchFacts(exact_block_dim=64),
    )

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert "verified non-cluster launch" in plan.unsupported.message


@pytest.mark.parametrize(
    ("group", "facts"),
    [
        (
            this_cluster(),
            LaunchFacts(
                exact_block_dim=64,
                exact_cluster_dim=2,
                cluster_launch=True,
            ),
        ),
        (
            this_grid(),
            LaunchFacts(
                exact_block_dim=64,
                exact_grid_dim=8,
                cooperative_launch=True,
            ),
        ),
    ],
)
def test_cluster_and_grid_require_backend_verified_launch_capabilities(group, facts):
    plan = _plan(group, _reduce(), facts)

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert "verified" in plan.unsupported.message


@pytest.mark.parametrize("group", [this_cluster(), this_grid()])
def test_cluster_launch_with_runtime_selected_shape_is_not_specialized(group):
    facts = LaunchFacts(
        exact_block_dim=64,
        exact_grid_dim=8 if group.kind == "grid" else None,
        cluster_launch=True,
        cooperative_launch=True if group.kind == "grid" else None,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
            *(
                (
                    LaunchFactOrigin(
                        "cooperative_launch",
                        "launch_config",
                        verified=True,
                    ),
                )
                if group.kind == "grid"
                else ()
            ),
        ),
    )

    plan = _plan(group, _reduce(), facts)

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert "exact static cluster dimensions" in plan.unsupported.message

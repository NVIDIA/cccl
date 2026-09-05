# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    COMPLETE_WARP_GROUP_KINDS,
    MAPPED_GROUP_KINDS,
    PHYSICAL_GROUP_KINDS,
    THREAD_GROUP_KINDS,
    THREAD_LEVELS,
    GroupByMapping,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    make_thread_group,
    normalize_thread_level,
    resolve_thread_group,
    this_block,
    this_warp,
)
from cuda.coop._core import group_dispatch as _group_dispatch
from cuda.coop._core import thread_group as _thread_group


def test_thread_group_kind_sets_preserve_distinct_contracts():
    assert PHYSICAL_GROUP_KINDS <= THREAD_LEVELS
    assert THREAD_LEVELS is not PHYSICAL_GROUP_KINDS
    assert PHYSICAL_GROUP_KINDS <= THREAD_GROUP_KINDS
    assert MAPPED_GROUP_KINDS <= THREAD_GROUP_KINDS
    assert PHYSICAL_GROUP_KINDS.isdisjoint(MAPPED_GROUP_KINDS)
    assert THREAD_GROUP_KINDS == PHYSICAL_GROUP_KINDS | MAPPED_GROUP_KINDS
    assert MAPPED_GROUP_KINDS < COMPLETE_WARP_GROUP_KINDS
    assert COMPLETE_WARP_GROUP_KINDS <= THREAD_GROUP_KINDS
    assert frozenset(_thread_group._CPP_LEVEL_EXPR) == THREAD_LEVELS
    assert frozenset(_group_dispatch._THREAD_LEVEL_ORDER) == THREAD_LEVELS
    assert frozenset(_group_dispatch._MAPPED_PARENT_LEVEL) == MAPPED_GROUP_KINDS
    assert frozenset(_group_dispatch._MAPPED_PARENT_LEVEL.values()) <= THREAD_LEVELS


def _resolved_hierarchy(
    block_dim,
    *,
    grid_dim=None,
    cluster_dim=None,
):
    return ThreadHierarchy._resolved(
        block_dim=block_dim,
        grid_dim=grid_dim,
        cluster_dim=cluster_dim,
    )


def test_resolved_thread_hierarchy_normalizes_dimensions_and_identity():
    hierarchy = _resolved_hierarchy(
        block_dim=(8, 4),
        grid_dim=3,
        cluster_dim=(2, 1),
    )

    assert hierarchy.block_dim == (8, 4, 1)
    assert hierarchy.grid_dim == (3, 1, 1)
    assert hierarchy.cluster_dim == (2, 1, 1)
    assert hierarchy.block_thread_count == 32
    assert hierarchy.thread_count == 32
    assert ThreadGroup(kind="cluster", hierarchy=hierarchy).static_size == 64
    assert ThreadGroup(kind="grid", hierarchy=hierarchy).static_size == 192
    assert hierarchy.block_dim_token == "b8x4"
    assert hierarchy.symbol_suffix == "g3_c2_b8x4"
    assert hierarchy.semantic_key == (
        (8, 4, 1),
        (3, 1, 1),
        (2, 1, 1),
        False,
    )
    assert hierarchy.has_static_extents_for("block")
    assert hierarchy.has_static_extents_for("cluster")
    assert hierarchy.has_static_extents_for("grid")


def test_current_group_can_be_resolved_without_changing_its_backend_type():
    class BackendGroup(ThreadGroup):
        pass

    current = make_thread_group(
        "block",
        group_type=BackendGroup,
        scope="example.coop",
    )

    assert type(current) is BackendGroup
    assert current.is_current
    assert current.static_thread_count is None
    assert current.static_size is None
    assert current.block_dim_token == "current"
    assert current.symbol_suffix == "block_current"

    resolved = current.with_hierarchy(
        _resolved_hierarchy((8, 4)),
        source="inferred_launch",
    )

    assert type(resolved) is BackendGroup
    assert resolved.is_static
    assert resolved.block_dim == (8, 4, 1)
    assert resolved.group_thread_count == 32
    assert resolved.thread_count == 32
    assert resolved.source == "inferred_launch"
    assert resolved.block_dim_token == "b8x4"
    assert resolved.symbol_suffix == "block_b8x4"
    assert resolved.semantic_key != current.semantic_key
    explicit = BackendGroup(kind="block", hierarchy=resolved.hierarchy)
    assert explicit == resolved
    assert hash(explicit) == hash(resolved)
    assert explicit.semantic_key == resolved.semantic_key
    core = ThreadGroup(kind="block", hierarchy=resolved.hierarchy)
    assert core != resolved


def test_core_this_helpers_only_build_current_launch_groups():
    block = this_block()
    warp = this_warp()

    assert block.is_current
    assert block.static_size is None
    assert warp.is_current
    assert warp.static_size == 32
    assert warp.thread_count == 32

    with pytest.raises(TypeError):
        this_block((16, 2))
    with pytest.raises(TypeError):
        this_warp(block_dim=64)


def test_hierarchy_does_not_invent_grid_facts_and_group_extents_are_distinct():
    hierarchy = _resolved_hierarchy(64)
    current_warp = this_warp()

    assert hierarchy.grid_dim is None
    assert hierarchy.block_thread_count == 64
    assert ThreadGroup(kind="block", hierarchy=hierarchy).static_size == 64
    assert ThreadGroup(kind="warp", hierarchy=hierarchy).static_size == 32
    assert ThreadGroup(kind="thread", hierarchy=hierarchy).static_size == 1
    assert current_warp.static_size == 32
    assert not current_warp.is_static


def test_this_warp_rejects_all_explicit_launch_metadata():
    with pytest.raises(TypeError):
        this_warp(16)
    with pytest.raises(TypeError):
        this_warp(block_dim=48)


def test_group_by_threads_within_warp_records_static_membership():
    parent = this_warp()
    group = parent.group_by(12, exhaustive=False)

    assert group.kind == "threads_within_warp"
    assert group.parent is parent
    assert group.mapping == GroupByMapping(
        unit="thread",
        parent="warp",
        count=12,
        exhaustive=False,
        synchronizer="lane",
    )
    assert group.static_size == 12
    assert group.groups_per_parent == 2
    assert group.remainder_count == 8
    assert group.complete_membership is False
    assert group.symbol_suffix == "threads_within_warp_12_partial_current"


def test_group_by_warps_within_block_resolves_implicit_parent_hierarchy():
    current = this_block().group_by(3, exhaustive=False)

    assert current.kind == "warps_within_block"
    assert current.static_size == 96
    assert current.groups_per_parent is None
    assert current.remainder_count is None
    assert current.complete_membership is None

    resolved = current.with_hierarchy(_resolved_hierarchy(320))
    assert resolved.parent is not None
    assert resolved.parent.hierarchy is resolved.hierarchy
    assert resolved.groups_per_parent == 3
    assert resolved.remainder_count == 1
    assert resolved.complete_membership is False
    assert resolved.semantic_key != current.semantic_key


def test_group_by_exhaustive_validation_and_nested_mapping_rejection():
    exhaustive = this_warp().group_by(8)
    assert exhaustive.groups_per_parent == 4
    assert exhaustive.complete_membership is True

    with pytest.raises(ValueError, match="requires the count to divide"):
        this_warp().group_by(12)

    incomplete_block = this_block().group_by(1)
    with pytest.raises(NotImplementedError, match="complete 32-thread warps"):
        resolve_thread_group(
            incomplete_block,
            LaunchFacts(exact_block_dim=48),
        ).require_supported()

    oversized = this_block().group_by(3, exhaustive=False)
    with pytest.raises(NotImplementedError, match="exceeds the resolved parent"):
        resolve_thread_group(
            oversized,
            LaunchFacts(exact_block_dim=64),
        ).require_supported()
    with pytest.raises(NotImplementedError, match="nested"):
        exhaustive.group_by(2)


@pytest.mark.parametrize("count", [True, 0, -1, 33, 1.5])
def test_group_by_rejects_invalid_thread_counts(count):
    with pytest.raises((TypeError, ValueError)):
        this_warp().group_by(count)


@pytest.mark.parametrize(
    "shape",
    [(), (1, 1, 1, 1), (4, 0), True, "32"],
)
def test_thread_hierarchy_rejects_all_explicit_block_shapes(shape):
    with pytest.raises(TypeError):
        ThreadHierarchy(block_dim=shape)


def test_thread_group_rejects_invalid_level():
    with pytest.raises(ValueError, match="level must be one of"):
        make_thread_group("tile")
    assert (
        normalize_thread_level(
            "gpu_thread",
            scope="cuda.coop",
            feature="test",
        )
        == "thread"
    )

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import cuda.coop._core.thread_group as _thread_group
from cuda.coop._core import (
    ThreadGroup,
    ThreadHierarchy,
    make_thread_group,
    this_block,
    this_warp,
)


def test_resolved_hierarchy_counts_threads_at_each_level():
    hierarchy = ThreadHierarchy._resolved(
        block_dim=(8, 4), grid_dim=3, cluster_dim=2
    )

    assert ThreadGroup(kind="block", hierarchy=hierarchy).static_size == 32
    assert ThreadGroup(kind="cluster", hierarchy=hierarchy).static_size == 64
    assert ThreadGroup(kind="grid", hierarchy=hierarchy).static_size == 192


def test_current_group_resolution_preserves_backend_type_and_cache_identity():
    class BackendGroup(ThreadGroup):
        pass

    current = make_thread_group("block", group_type=BackendGroup)
    assert current.static_size is None

    hierarchy = ThreadHierarchy._resolved(block_dim=(8, 4))
    resolved = current.with_hierarchy(hierarchy, source="inferred_launch")
    assert type(resolved) is BackendGroup
    assert resolved.static_size == 32
    assert current not in {resolved}
    assert BackendGroup(kind="block", hierarchy=hierarchy) in {resolved}


def test_physical_group_rendering_uses_current_or_resolved_hierarchy():
    current = this_block()
    assert "implicit_hierarchy()" in _thread_group.render_group_decl(current)

    hierarchy = ThreadHierarchy._resolved(block_dim=(8, 4))
    resolved = current.with_hierarchy(hierarchy)
    assert "block_dims<8, 4>()" in "\n".join(
        _thread_group.render_hierarchy_decl(hierarchy)
    )
    assert "this_block group{hierarchy}" in _thread_group.render_group_decl(
        resolved
    )


@pytest.mark.parametrize(
    ("kind", "count", "exhaustive", "synchronizer"),
    [
        ("warp", 8, True, "lane_synchronizer{}"),
        ("block", 2, False, "barrier_synchronizer{group_barriers}"),
    ],
)
def test_mapped_group_rendering_uses_membership_and_synchronizer(
    kind, count, exhaustive, synchronizer
):
    group = ThreadGroup(
        kind=kind, hierarchy=ThreadHierarchy._resolved(block_dim=160)
    ).group_by(count, exhaustive=exhaustive)
    source = "\n".join(_thread_group.render_group_decl_lines(group))

    assert f"group_by<{count}, {str(exhaustive).lower()}>" in source
    assert synchronizer in source
    if kind == "block":
        assert "barrier<::cuda::thread_scope_block>[2]" in source


def test_partial_warp_mapping_excludes_remainder_lanes():
    group = this_warp().group_by(12, exhaustive=False)

    assert group.static_size == 12
    assert group.groups_per_parent == 2
    assert group.remainder_count == 8


def test_block_mapping_waits_for_launch_dimensions():
    current = this_block().group_by(3, exhaustive=False)
    assert current.static_size == 96
    assert current.groups_per_parent is None

    resolved = current.with_hierarchy(ThreadHierarchy._resolved(block_dim=320))
    assert resolved.groups_per_parent == 3
    assert resolved.remainder_count == 1


@pytest.mark.parametrize(
    ("block_threads", "count", "exhaustive", "message"),
    [
        (48, 1, False, "complete warps"),
        (64, 3, False, "cannot exceed the parent warp count"),
        (128, 3, True, "requires the count to divide"),
    ],
)
def test_group_by_revalidates_deferred_constraints(
    block_threads, count, exhaustive, message
):
    current = this_block().group_by(count, exhaustive=exhaustive)

    with pytest.raises(ValueError, match=message):
        current.with_hierarchy(
            ThreadHierarchy._resolved(block_dim=block_threads)
        )


def test_group_by_requires_exhaustive_non_nested_membership():
    with pytest.raises(ValueError, match="requires the count to divide"):
        this_warp().group_by(12)
    with pytest.raises(NotImplementedError, match="nested"):
        this_warp().group_by(8).group_by(2)

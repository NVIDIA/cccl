# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import cuda.coop.numba_mlir as coop
import cuda.coop.numba_mlir._thread_group as numba_mlir_groups
from cuda.coop._core import ThreadHierarchy as CoreThreadHierarchy


def test_group_exports_use_the_shared_hierarchy_contract():
    assert coop.Hierarchy is coop.ThreadHierarchy
    assert coop.ThreadHierarchy is CoreThreadHierarchy

    for name in (
        "Hierarchy",
        "ThreadGroup",
        "ThreadHierarchy",
        "this_thread",
        "this_warp",
        "this_block",
        "this_cluster",
        "this_grid",
    ):
        assert name in coop.__all__


def test_current_group_construction_preserves_backend_type():
    current = coop.this_block()

    assert type(current) is coop.ThreadGroup
    assert current.is_current
    assert repr(current).startswith("ThreadGroup(kind='block'")
    assert current.block_dim is None
    assert current.static_size is None
    assert coop.this_cluster().static_size is None
    assert coop.this_grid().static_size is None
    assert coop.this_thread().static_size == 1
    assert coop.this_warp().static_size == 32


def test_group_equality_hashing_and_group_by_use_numba_mlir_type():
    first = coop.this_block()
    second = coop.this_block()
    mapped = first.group_by(2)

    assert first == second
    assert hash(first) == hash(second)
    assert type(mapped) is coop.ThreadGroup
    assert type(mapped.parent) is coop.ThreadGroup
    assert mapped.kind == "warps_within_block"
    assert mapped.static_size == 64
    assert mapped.groups_per_parent is None
    assert mapped.complete_membership is None


def test_group_constructors_preserve_shared_validation():
    with pytest.raises(TypeError):
        coop.ThreadHierarchy(block_dim=64)
    with pytest.raises(TypeError):
        coop.this_warp(16)
    with pytest.raises(TypeError):
        coop.this_warp(block_dim=64)
    with pytest.raises(ValueError, match="requires the count to divide"):
        coop.this_warp().group_by(12)
    with pytest.raises(NotImplementedError, match="nested"):
        coop.this_warp().group_by(8).group_by(2)


def test_group_methods_use_one_compile_time_marker(monkeypatch):
    calls = []
    marker_result = object()

    def marker(group, operation, *args):
        calls.append((group, operation, args))
        return marker_result

    monkeypatch.setattr(
        numba_mlir_groups,
        "_thread_group_method_marker",
        marker,
    )
    group = coop.this_block()

    assert group.rank("block") is marker_result
    assert group.count("grid") is marker_result
    assert group.rank_as("uint32", "warp") is marker_result
    assert group.count_as("uint64", "thread") is marker_result
    assert group.sync() is None
    assert group.sync_aligned() is None
    assert group.is_member() is marker_result
    assert calls == [
        (group, "rank", (None, "block")),
        (group, "count", (None, "grid")),
        (group, "rank", ("uint32", "warp")),
        (group, "count", ("uint64", "thread")),
        (group, "sync", ()),
        (group, "sync_aligned", ()),
        (group, "is_member", ()),
    ]


def test_group_method_marker_fails_clearly_outside_compilation():
    with pytest.raises(RuntimeError, match="whole-function planner"):
        coop.this_block().rank()

    with pytest.raises(ValueError, match="level must be one of"):
        coop.this_block().count("tile")

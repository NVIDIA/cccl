# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import cuda.coop as portable_coop
import cuda.coop.numba_mlir as coop
from cuda.coop._core import ThreadHierarchy as CoreThreadHierarchy

_UNSUPPORTED_THREAD_GROUP_METHODS = frozenset(
    {
        "count",
        "count_as",
        "is_member",
        "rank",
        "rank_as",
        "sync",
        "sync_aligned",
    }
)


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


@pytest.mark.parametrize("api", (portable_coop, coop), ids=("portable", "qualified"))
def test_thread_group_runtime_surface_is_descriptor_only(api):
    group = api.this_block()

    assert callable(group.group_by)
    assert _UNSUPPORTED_THREAD_GROUP_METHODS.isdisjoint(dir(group))
    assert all(
        not hasattr(group, method) for method in _UNSUPPORTED_THREAD_GROUP_METHODS
    )

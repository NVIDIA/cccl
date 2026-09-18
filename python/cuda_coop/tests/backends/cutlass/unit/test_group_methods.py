# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Exact launch requirements and storage-free hierarchy metadata."""

import pytest

cutlass = pytest.importorskip("cutlass")

from types import SimpleNamespace

from cuda.coop._core import LaunchFacts
from cuda.coop.cutlass import (
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from cuda.coop.cutlass._compiler import _rendering
from cuda.coop.cutlass._compiler._launch import launch_facts_from_cutlass_api
from cuda.coop.cutlass._lowering import _thread_group as lowering

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _facts(monkeypatch, **values):
    facts = launch_facts_from_cutlass_api(SimpleNamespace(**values))
    monkeypatch.setattr(lowering, "current_kernel_launch_facts", lambda: facts)
    return facts


@pytest.mark.parametrize("op", ("rank", "count", "is_member"))
def test_mapped_metadata_has_no_storage(monkeypatch, op):
    _facts(monkeypatch, exact_block_dim=(8, 4, 3))
    group = lowering._resolve_method_group(
        this_block().group_by(2, exhaustive=False), op
    )
    request = lowering._CudaxGroupRequest(group, op, result_type=cutlass.Uint32)
    source = _rendering.render_bundle_source([request])
    assert "__shared__" not in source
    assert "barriers_storage" not in source
    assert "__syncthreads" not in source
    assert "group.sync" not in source
    assert not _rendering.bundle_scratch_layout_probes([request])


@pytest.mark.parametrize("op", ("rank", "count", "is_member", "sync"))
def test_thread_needs_no_block(monkeypatch, op):
    _facts(monkeypatch)
    assert lowering._resolve_method_group(this_thread(), op).kind == "thread"


@pytest.mark.parametrize("block", ((16, 1, 1), (24, 1, 1), (48, 1, 1)))
def test_warp_needs_complete_partition(monkeypatch, block):
    _facts(monkeypatch, exact_block_dim=block)
    with pytest.raises(NotImplementedError, match="complete"):
        lowering._resolve_method_group(this_warp(), "rank")


def test_block_query_allows_partial_warp(monkeypatch):
    _facts(monkeypatch, exact_block_dim=(48, 1, 1))
    group = lowering._resolve_method_group(this_block(), "count", "warp")
    assert group.hierarchy.block_thread_count == 48


@pytest.mark.parametrize("constructor", (this_cluster, this_grid))
def test_cluster_mode_must_be_verified(monkeypatch, constructor):
    facts = LaunchFacts(
        exact_block_dim=(64, 1, 1),
        exact_grid_dim=(2, 1, 1),
        exact_cluster_dim=(1, 1, 1),
        cluster_launch=False,
    )
    monkeypatch.setattr(lowering, "current_kernel_launch_facts", lambda: facts)
    with pytest.raises(NotImplementedError, match="verified"):
        lowering._resolve_method_group(constructor(), "count")


def test_grid_counts_clusters(monkeypatch):
    _facts(
        monkeypatch,
        exact_block_dim=(8, 4, 2),
        exact_grid_dim=(4, 6, 2),
        exact_cluster_dim=(2, 3, 1),
        cluster_launch=True,
    )
    group = lowering._resolve_method_group(this_grid(), "count")
    assert group.hierarchy.grid_dim == (2, 2, 2)
    assert group.hierarchy.cluster_dim == (2, 3, 1)


def test_noncluster_unit_inference(monkeypatch):
    _facts(
        monkeypatch,
        exact_block_dim=(32, 1, 1),
        exact_grid_dim=(2, 1, 1),
        cluster_launch=False,
    )
    group = lowering._resolve_method_group(this_cluster(), "count")
    assert group.hierarchy.cluster_dim == (1, 1, 1)


def test_grid_requires_divisibility(monkeypatch):
    _facts(
        monkeypatch,
        exact_block_dim=(32, 1, 1),
        exact_grid_dim=(3, 1, 1),
        exact_cluster_dim=(2, 1, 1),
        cluster_launch=True,
    )
    with pytest.raises(NotImplementedError, match="divisible"):
        lowering._resolve_method_group(this_grid(), "count")


def test_query_defaults():
    assert lowering._result_type(this_thread(), "thread", None) is cutlass.Uint32
    assert lowering._result_type(this_block(), "thread", None) is cutlass.Uint32
    assert lowering._result_type(this_block(), "grid", None) is cutlass.Uint64
    assert lowering._result_type(this_grid(), "thread", None) is cutlass.Uint64


def test_query_registration_rolls_back(monkeypatch):
    events = []
    sentinel = object()
    monkeypatch.setattr(
        lowering._state, "snapshot_active_session_state", lambda: sentinel
    )
    monkeypatch.setattr(lowering._state, "register_request", events.append)

    def restore(snapshot):
        assert snapshot is sentinel
        events.clear()

    monkeypatch.setattr(lowering._state, "restore_active_session_state", restore)

    def failure(**kwargs):
        raise ValueError("bad ffi")

    monkeypatch.setattr(lowering, "ffi", failure)
    request = lowering._CudaxGroupRequest(
        this_thread(), "rank", result_type=cutlass.Uint32
    )
    with pytest.raises(ValueError, match="bad ffi"):
        lowering._emit(request, cutlass.Uint32)
    assert not events

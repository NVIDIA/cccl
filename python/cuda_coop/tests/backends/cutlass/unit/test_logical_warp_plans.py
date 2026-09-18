# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Logical Warp plans preserve subgroup participation and independent tiles."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreKind,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_warp,
)
from cuda.coop.cutlass._compiler import _rendering
from cuda.coop.cutlass._lowering import _load_store

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]

_WIDTHS = (1, 2, 4, 8, 16, 32)
_ALGORITHMS = ("direct", "striped", "vectorize", "transpose")


def _request(
    width,
    *,
    kind="load",
    algorithm="direct",
    block=(8, 4, 2),
    exhaustive=True,
    valid=None,
    offset=None,
    group=None,
):
    return _load_store._make_request(
        group=group
        if group is not None
        else this_warp().group_by(width, exhaustive=exhaustive),
        launch=LaunchFacts(exact_block_dim=block),
        kind=GroupLoadStoreKind(kind),
        value_type=cutlass.Int32,
        items_per_thread=2,
        algorithm=algorithm,
        valid_items_binding=valid or ArgumentBinding.omitted(),
        oob_default_binding=ArgumentBinding.omitted(),
        offset_binding=offset or ArgumentBinding.omitted(),
    )


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("kind", ("load", "store"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("exhaustive", (True, False))
def test_logical_group_contract(width, kind, algorithm, exhaustive):
    request = _request(width, kind=kind, algorithm=algorithm, exhaustive=exhaustive)
    plan = request.plan
    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.resolved_group.static_size == width
    assert plan.resolved_group.complete_membership
    assert plan.topology.logical_width == width
    assert plan.topology.instances == 64 // width
    assert request.group_instances == 64 // width
    assert _load_store._required_static_elements(request) == 128
    if algorithm == "transpose":
        assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
        assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
        assert len(_rendering.bundle_scratch_layout_probes([request])) == 1
    else:
        assert plan.temp_storage.ownership is StorageOwnership.NONE
        assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE
        assert not _rendering.bundle_scratch_layout_probes([request])


@pytest.mark.parametrize("width", _WIDTHS)
def test_subgroup_tile_bounds(width):
    tile_items = 2 * width
    valid_items = tile_items - 1
    request = _request(
        width,
        valid=ArgumentBinding.static(valid_items),
        offset=ArgumentBinding.static(5),
    )
    assert (
        _load_store._required_static_elements(request)
        == 5 + 128 - tile_items + valid_items
    )
    with pytest.raises(ValueError, match="valid_items"):
        _request(width, valid=ArgumentBinding.static(tile_items + 1))


@pytest.mark.parametrize("width", _WIDTHS)
def test_subgroup_offset_headroom(width):
    maximum = (1 << 63) - 1 - (128 - 2 * width)
    assert (
        _request(width, offset=ArgumentBinding.static(maximum)).operation.offset.value
        == maximum
    )
    with pytest.raises(ValueError, match="offset"):
        _request(width, offset=ArgumentBinding.static(maximum + 1))


@pytest.mark.parametrize("width", _WIDTHS)
def test_scratch_identity_counts_groups(width):
    one_warp = _request(width, algorithm="transpose", block=(32, 1, 1))
    two_warps = _request(width, algorithm="transpose")
    assert one_warp.implementation.semantic_key == two_warps.implementation.semantic_key
    assert one_warp.scratch_requirement_key != two_warps.scratch_requirement_key
    assert one_warp.symbol_name != two_warps.symbol_name
    assert len(_rendering.bundle_scratch_layout_probes([one_warp, two_warps])) == 2


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_width_three_rejected(algorithm):
    with pytest.raises(NotImplementedError, match="power-of-two group width"):
        _request(3, algorithm=algorithm, exhaustive=False)


def test_partial_physical_warp_rejected():
    with pytest.raises(NotImplementedError, match="complete"):
        _request(8, block=(8, 3, 1))


def test_unknown_block_rejected():
    with pytest.raises(NotImplementedError, match="exact block dimensions"):
        _request(8, block=None)


def test_mapped_warps_rejected():
    with pytest.raises(NotImplementedError, match="Load/Store supports"):
        _request(32, group=this_block().group_by(1))


@pytest.mark.parametrize("algorithm", ("warp_transpose", "warp_transpose_timesliced"))
def test_block_algorithms_rejected(algorithm):
    with pytest.raises(NotImplementedError, match="algorithm"):
        _request(8, algorithm=algorithm)

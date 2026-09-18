# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Physical Warp provider plans retain group-local storage and tile bounds."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreKind,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    this_warp,
)
from cuda.coop.cutlass._compiler import _rendering
from cuda.coop.cutlass._lowering import _load_store

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(
    *,
    algorithm="direct",
    kind="load",
    block=(64, 1, 1),
    valid=None,
    offset=None,
):
    return _load_store._make_request(
        group=this_warp(),
        launch=LaunchFacts(exact_block_dim=block),
        kind=GroupLoadStoreKind(kind),
        value_type=cutlass.Int32,
        items_per_thread=2,
        algorithm=algorithm,
        valid_items_binding=valid or ArgumentBinding.omitted(),
        oob_default_binding=ArgumentBinding.omitted(),
        offset_binding=offset or ArgumentBinding.omitted(),
    )


@pytest.mark.parametrize("algorithm", ("direct", "striped", "vectorize"))
@pytest.mark.parametrize("kind", ("load", "store"))
def test_warp_storage_free_contract(algorithm, kind):
    request = _request(algorithm=algorithm, kind=kind)
    assert request.plan.target is GroupLoweringTarget.CUB_WARP
    assert request.plan.topology.instances == 2
    assert request.plan.topology.logical_width == 32
    assert request.plan.temp_storage.ownership is StorageOwnership.NONE
    assert (
        request.plan.synchronization.storage_reuse_barrier is SynchronizationScope.NONE
    )
    assert not _rendering.bundle_scratch_layout_probes([request])
    source = _rendering.render_bundle_source([request])
    assert "TempStorage" not in source
    assert "temp_storage_smem_addr" not in source
    assert "__shared__" not in source
    assert "__syncwarp" not in source
    assert "__syncthreads" not in source


def test_scratch_probe_counts_instances():
    one = _request(algorithm="transpose", block=(32, 1, 1))
    two = _request(algorithm="transpose", block=(8, 4, 2))
    assert one.implementation.semantic_key == two.implementation.semantic_key
    assert one.scratch_requirement_key != two.scratch_requirement_key
    assert one.symbol_name != two.symbol_name
    probes = _rendering.bundle_scratch_layout_probes([one, two])
    assert len(probes) == 2
    first, second = (probes[request.scratch_requirement_key] for request in (one, two))
    assert first.size_expression != second.size_expression
    assert two.plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert two.plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
    source = _rendering.render_bundle_source([two])
    assert "__syncthreads" not in source
    assert "bar.sync" not in source


@pytest.mark.parametrize("valid", (0, 17, 64))
@pytest.mark.parametrize("algorithm", ("direct", "striped", "vectorize", "transpose"))
def test_extent_includes_all_group_tiles(valid, algorithm):
    request = _request(
        algorithm=algorithm,
        block=(8, 4, 3),
        valid=ArgumentBinding.static(valid),
        offset=ArgumentBinding.static(5),
    )
    assert _load_store._required_static_elements(request) == 5 + 2 * 64 + valid


def test_static_and_runtime_extent():
    assert _load_store._required_static_elements(_request()) == 128
    assert (
        _load_store._required_static_elements(
            _request(offset=ArgumentBinding.runtime())
        )
        is None
    )


@pytest.mark.parametrize("valid", (-1, 65, 128))
def test_valid_count_is_group_local(valid):
    with pytest.raises(ValueError, match="valid_items"):
        _request(valid=ArgumentBinding.static(valid))


def test_offset_headroom():
    maximum = (1 << 63) - 1 - 64
    request = _request(offset=ArgumentBinding.static(maximum))
    assert request.operation.offset.value == maximum
    with pytest.raises(ValueError, match="offset"):
        _request(offset=ArgumentBinding.static(maximum + 1))


@pytest.mark.parametrize("block", ((1, 1, 1), (31, 1, 1), (33, 1, 1), (8, 3, 1)))
def test_partial_warp_block_rejected(block):
    with pytest.raises(NotImplementedError, match="complete"):
        _request(block=block)


def test_unknown_block_rejected():
    with pytest.raises(NotImplementedError, match="exact block dimensions"):
        _request(block=None)


@pytest.mark.parametrize("algorithm", ("warp_transpose", "warp_transpose_timesliced"))
def test_block_algorithm_rejected(algorithm):
    with pytest.raises(NotImplementedError, match="algorithm"):
        _request(algorithm=algorithm)

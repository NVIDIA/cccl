# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load/Store providers retain the shared planner's storage contracts."""

import pytest

pytest.importorskip("cutlass")

from cutlass import Int32

from cuda.coop._core import ArgumentBinding, StorageOwnership, this_block
from cuda.coop.cutlass._compiler import _rendering
from cuda.coop.cutlass._lowering._load_store import _CubLoadStoreRequest
from tests.support.group_planning import _load_store, _plan

pytestmark = [pytest.mark.unit, pytest.mark.backend_cutlass]


def _request(kind, algorithm, **kwargs):
    return _CubLoadStoreRequest(
        _plan(
            this_block(), _load_store(kind, dtype=Int32, algorithm=algorithm, **kwargs)
        ),
        Int32,
    )


@pytest.mark.parametrize("algorithm", ("direct", "striped", "vectorize"))
@pytest.mark.parametrize("kind", ("load", "store"))
def test_storage_free_provider_ignores_descriptor_controls(kind, algorithm):
    implicit = _request(kind, algorithm)
    explicit = _request(
        kind,
        algorithm,
        storage_ownership=StorageOwnership.CALLER,
        storage_sharing="exclusive",
        storage_size_in_bytes=1,
        storage_alignment=64,
        storage_auto_sync=False,
    )
    assert implicit == explicit
    assert implicit.symbol_name == explicit.symbol_name
    source = _rendering.render_bundle_source([implicit])
    assert source == _rendering.render_bundle_source([explicit])
    assert not _rendering.bundle_scratch_layout_probes([explicit])
    for token in ("temp_storage", "TempStorage", "__shared__", "__syncthreads"):
        assert token not in source


@pytest.mark.parametrize(
    "algorithm", ("transpose", "warp_transpose", "warp_transpose_timesliced")
)
def test_partial_transpose_uses_shared_preservation_specialization(algorithm):
    load = _request("load", algorithm, valid_items=ArgumentBinding.runtime())
    store = _request("store", algorithm)
    source = _rendering.render_bundle_source([load, store, load])
    assert source.count("class CudaCoopBlockLoadPreservingInvalid") == 1
    assert (
        "using implementation_type = ::cub::CudaCoopBlockLoadPreservingInvalid<"
        in source
    )
    assert "result_items[0], result_items[1]" in source
    probes = _rendering.bundle_scratch_layout_probes([load, store, load])
    assert set(probes) == {load.scratch_requirement_key, store.scratch_requirement_key}
    for probe in probes.values():
        assert "TempStorage" in probe.size_expression
        assert "TempStorage" in probe.alignment_expression
    assert source.index("namespace cub") < source.index('extern "C"')

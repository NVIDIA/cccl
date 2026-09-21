# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Neighbor calls preserve the shared planner's types and storage contract."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_warp,
)
from cuda.coop.cutlass import TempStorage, ThreadData
from cuda.coop.cutlass._compiler import _rendering, _state, _storage
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES
from cuda.coop.cutlass._lowering import _neighbors

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(**options):
    arguments = dict(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        dtype=cutlass.Int32,
        items=3,
        operation="adjacent_difference",
        mode="left",
    )
    arguments.update(options)
    return _neighbors._CubNeighborRequest(_neighbors._make_neighbor_plan(**arguments))


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
@pytest.mark.parametrize(
    "operation,mode",
    [
        ("adjacent_difference", "left"),
        ("adjacent_difference", "right"),
        ("discontinuity", "heads"),
        ("discontinuity", "tails"),
        ("discontinuity", "heads_and_tails"),
    ],
)
def test_results(dtype, operation, mode):
    request = _request(dtype=dtype, operation=operation, mode=mode)
    assert request.output_type is (
        dtype if operation == "adjacent_difference" else cutlass.Int32
    )
    assert len(request.plan.result.values) == (2 if mode == "heads_and_tails" else 1)
    assert request.plan.temp_storage.exact_layout_required
    source = _rendering.render_bundle_source([request])
    assert source.index("struct CudaCoopBlock") < source.index('extern "C"')


@pytest.mark.parametrize("count", (-1, 193, 1 << 32, True, 1.5, np.uint64(5)))
def test_invalid_count(count):
    # NumPy integers are trace-static; uint64 is valid when its value fits.
    if isinstance(count, np.uint64):
        assert _request(valid_items=count).plan.call.operation.valid_items.value == 5
    else:
        with pytest.raises((TypeError, ValueError), match="valid_items"):
            _request(valid_items=count)


@pytest.mark.parametrize(
    "options",
    [
        dict(mode="right", predecessor=True),
        dict(successor=True),
        dict(mode="right", valid_items=1, successor=True),
        dict(operation="discontinuity", mode="heads", valid_items=1),
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        _request(**options)


def test_complete_block_required():
    with pytest.raises((ValueError, NotImplementedError), match="block"):
        _request(group=this_warp())


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_storage_controls(sharing, auto_sync):
    request = _request(
        temp_storage=TempStorage(alignment=128, sharing=sharing, auto_sync=auto_sync)
    )
    assert request.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert request.plan.temp_storage.requested_alignment == 128
    assert request.plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


def test_semantics_and_storage_affect_identity():
    requests = [
        _request(),
        _request(mode="right"),
        _request(valid_items=0),
        _request(predecessor=True),
        _request(temp_storage=TempStorage(auto_sync=False)),
    ]
    assert len({request.symbol_name for request in requests}) == len(requests)


def test_mismatched_result_rejected():
    request = _request()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], dtype=cutlass.Float64),),
    )
    with pytest.raises(ValueError, match="result"):
        _neighbors._CubNeighborRequest(replace(request.plan, result=result))


def test_failed_storage_emission_restores_session(monkeypatch):
    saved, restored, requests = object(), [], []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "register_request", requests.append)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(
        _neighbors, "_make_rmem_tensor", lambda *args: SimpleNamespace()
    )

    def fail(*args, **kwargs):
        raise RuntimeError("scratch emission failed")

    monkeypatch.setattr(_storage, "register_deferred_temp_storage_event", fail)
    values = ThreadData(2, dtype=cutlass.Int32, values=[3, 1])
    with pytest.raises(RuntimeError, match="scratch emission failed"):
        _neighbors.provider_neighbors(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
            values=values,
            operation="adjacent_difference",
            mode="left",
            valid_items=None,
            tile_predecessor_item=None,
            tile_successor_item=None,
            temp_storage=None,
        )
    assert len(requests) == 1
    assert restored == [saved]
    assert values.values("adjacent_difference") == (3, 1)

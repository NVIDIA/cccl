# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Histogram planning includes counter storage and independent output types."""

from dataclasses import replace
from types import SimpleNamespace

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
from cuda.coop.cutlass._lowering import _histogram

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(**options):
    arguments = dict(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        sample_type=cutlass.Int32,
        items=3,
        bins=65,
        bins_per_thread=2,
        counter_type=cutlass.Int64,
        algorithm="atomic",
    )
    arguments.update(options)
    return _histogram._CubHistogramRequest(_histogram._make_histogram_plan(**arguments))


@pytest.mark.parametrize("sample", tuple(_histogram._SAMPLES))
@pytest.mark.parametrize("counter", tuple(_histogram._COUNTERS))
@pytest.mark.parametrize("algorithm", ("atomic", "sort"))
def test_types_and_result(sample, counter, algorithm):
    request = _request(sample_type=sample, counter_type=counter, algorithm=algorithm)
    assert request.plan.result.values[0].dtype is counter
    assert request.plan.result.values[0].items_per_member == 2
    probe = _histogram._scratch_probe(request)
    assert "BlockHistogramCoop" in probe.size_expression
    source = _rendering.render_bundle_source([request])
    assert source.index("class BlockHistogramCoop") < source.index('extern "C"')
    assert "internal_counter_t counters[Bins]" in source


@pytest.mark.parametrize(
    "options",
    [
        dict(sample_type=cutlass.Float32),
        dict(counter_type=cutlass.Uint8),
        dict(bins=0),
        dict(bins=129),
        dict(bins_per_thread=0),
        dict(bins=True),
        dict(algorithm="other"),
        dict(group=this_warp()),
        dict(launch=LaunchFacts(exact_block_dim=(8, 8, 1))),
    ],
)
def test_invalid_contract(options):
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        _request(**options)


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_storage_controls(sharing, auto_sync):
    request = _request(
        temp_storage=TempStorage(sharing=sharing, auto_sync=auto_sync, alignment=1)
    )
    assert request.plan.temp_storage.exact_layout_required
    assert request.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert request.plan.temp_storage.requested_alignment == 1
    assert request.plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


def test_output_and_algorithm_affect_identity():
    requests = [
        _request(),
        _request(counter_type=cutlass.Int32),
        _request(algorithm="sort"),
        _request(bins=64),
        _request(bins_per_thread=3),
    ]
    assert len({request.symbol_name for request in requests}) == len(requests)
    assert len({request.scratch_requirement_key for request in requests}) == len(
        requests
    )


def test_mismatched_result_rejected():
    request = _request()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], items_per_member=3),),
    )
    with pytest.raises(ValueError, match="result"):
        _histogram._CubHistogramRequest(replace(request.plan, result=result))


def test_failed_storage_emission_restores_session(monkeypatch):
    saved, restored = object(), []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "register_request", lambda request: None)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(
        _histogram, "_make_rmem_tensor", lambda *args: SimpleNamespace()
    )

    def fail(*args, **kwargs):
        raise RuntimeError("scratch emission failed")

    monkeypatch.setattr(_storage, "register_deferred_temp_storage_event", fail)
    samples = ThreadData(2, dtype=cutlass.Int32, values=[3, 1])
    with pytest.raises(RuntimeError, match="scratch emission failed"):
        _histogram.provider_histogram(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
            samples=samples,
            bins=5,
            bins_per_thread=1,
            counter_dtype=None,
            algorithm="atomic",
            temp_storage=None,
        )
    assert restored == [saved]
    assert samples.values("histogram") == (3, 1)

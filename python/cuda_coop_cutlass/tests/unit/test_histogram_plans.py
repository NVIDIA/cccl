# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check Histogram plans, scratch policy, and queued session restoration.

Sample and counter types are independent. Plans also distinguish the
number of bins, per-thread output extent, and algorithm. These tests
inspect generated wrappers and plan metadata without launching kernels.
A failure during scratch registration must restore saved session state.
"""

from dataclasses import replace
from types import SimpleNamespace

import cutlass
import pytest

from cuda.coop.cutlass import (
    TempStorage,
    ThreadData,
    histogram_composite,
    histogram_init,
)
from cuda.coop.cutlass._compiler import _rendering, _state, _storage
from cuda.coop.cutlass._lowering import _histogram
from cuda.coop.cutlass._planning import (
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_warp,
)

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(**options):
    arguments = {
        "group": this_block(),
        "launch": LaunchFacts(exact_block_dim=(64, 1, 1)),
        "sample_type": cutlass.Int32,
        "items": 3,
        "bins": 65,
        "bins_per_thread": 2,
        "counter_type": cutlass.Int64,
        "algorithm": "atomic",
    }
    arguments.update(options)
    return _histogram._CubHistogramRequest(
        _histogram._make_histogram_plan(**arguments)
    )


@pytest.mark.parametrize("sample", tuple(_histogram._SAMPLES))
@pytest.mark.parametrize("counter", tuple(_histogram._COUNTERS))
@pytest.mark.parametrize("algorithm", ("atomic", "sort"))
def test_types_and_result(sample, counter, algorithm):
    request = _request(
        sample_type=sample, counter_type=counter, algorithm=algorithm
    )
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
        {"sample_type": cutlass.Float32},
        {"counter_type": cutlass.Uint8},
        {"bins": 0},
        {"bins": 129},
        {"bins_per_thread": 0},
        {"bins": True},
        {"algorithm": "other"},
        {"group": this_warp()},
        {"launch": LaunchFacts(exact_block_dim=(8, 8, 1))},
    ],
)
def test_invalid_contract(options):
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        _request(**options)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"bins_per_thread": 0}, "bins_per_thread must be a positive integer"),
        (
            {"bins": 1 << 31},
            "histogram bins_per_thread must provide capacity for every bin",
        ),
        (
            {},
            "histogram tile and output sizes must fit a signed 32-bit integer",
        ),
    ],
)
def test_one_shot_oversized_tile_diagnostics(options, message):
    with pytest.raises(ValueError) as error:
        _request(items=1 << 31, **options)
    assert str(error.value) == message


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_storage_controls(sharing, auto_sync):
    request = _request(
        temp_storage=TempStorage(
            sharing=sharing, auto_sync=auto_sync, alignment=1
        )
    )
    assert request.plan.temp_storage.exact_layout_required
    assert request.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert request.plan.temp_storage.requested_alignment == 1
    assert request.plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


def test_output_and_algorithm_affect_identity():
    """Give each output shape and algorithm its own wrapper and scratch key.

    Counter type, algorithm, bin count, and per-thread output extent each
    change the provider request. None may reuse a mismatched specialization.
    """

    requests = [
        _request(),
        _request(counter_type=cutlass.Int32),
        _request(algorithm="sort"),
        _request(bins=64),
        _request(bins_per_thread=3),
    ]
    assert len({request.symbol_name for request in requests}) == len(requests)
    assert len(
        {request.scratch_requirement_key for request in requests}
    ) == len(requests)


def test_mismatched_result_rejected():
    request = _request()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], items_per_member=3),),
    )
    with pytest.raises(ValueError, match="result"):
        _histogram._CubHistogramRequest(replace(request.plan, result=result))


@pytest.mark.parametrize("composite", (False, True))
def test_failed_storage_emission_restores_session(monkeypatch, composite):
    """Restore saved session state if Histogram scratch registration fails.

    Stubs replace request recording and register-tensor creation, so the call
    needs no CuTe kernel trace. With temp_storage=None, the provider still
    registers automatic scratch. The injected failure must restore the saved
    snapshot and leave the samples unchanged.
    """

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
    samples = ThreadData(items_per_thread=2, dtype=cutlass.Int32, values=[3, 1])
    with pytest.raises(RuntimeError, match="scratch emission failed"):
        common = dict(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
            samples=samples,
            bins=5,
            algorithm="atomic",
            temp_storage=None,
        )
        if composite:
            _histogram.provider_histogram_composite(
                **common, counters=SimpleNamespace(element_type=cutlass.Int32)
            )
        else:
            _histogram.provider_histogram(
                **common, bins_per_thread=1, counter_dtype=None
            )
    assert restored == [saved]
    assert samples.values("histogram") == (3, 1)


def _composite_request(
    *, algorithm="atomic", items=3, counter=cutlass.Int32, temp_storage=None
):
    operation = _histogram.GroupHistogramCompositeSemantics(
        cutlass.Uint8, items, 70, counter, algorithm
    )
    return _histogram._CubHistogramRequest(
        _histogram._plan_with_storage(
            this_block(),
            LaunchFacts(exact_block_dim=(64, 1, 1)),
            operation,
            temp_storage,
        )
    )


@pytest.mark.parametrize("algorithm", ("atomic", "sort"))
@pytest.mark.parametrize(
    "counter", (cutlass.Int32, cutlass.Uint32, cutlass.Int64, cutlass.Uint64)
)
def test_composite_shared_counters(algorithm, counter):
    request = _composite_request(algorithm=algorithm, counter=counter)
    assert request.plan.result is None
    assert request.plan.provenance.method == "Composite"
    assert (
        request.plan.synchronization.storage_reuse_barrier
        is SynchronizationScope.BLOCK
    )
    assert (
        "BlockHistogram<" in _histogram._scratch_probe(request).size_expression
    )
    source = _rendering.render_bundle_source([request])
    assert "unsigned int counters_address" in source
    assert "__cvta_shared_to_generic(counters_address)" in source
    assert ".Composite(samples, counters)" in source
    assert "if (storage_auto_sync != 0) { __syncthreads(); }" in source
    assert ".Histogram(" not in source
    assert "BlockHistogramCoop" not in source
    assert _request().symbol_name != request.symbol_name


def test_composite_sort_tile_limit():
    with pytest.raises(ValueError, match="65535"):
        _composite_request(items=1024, algorithm="sort")
    assert _composite_request(items=1024).plan.result is None


def test_histogram_init_requires_one_source_of_counters():
    block = this_block()
    with pytest.raises(TypeError, match="requires counters or bins"):
        histogram_init(block)
    with pytest.raises(TypeError, match="either counters or bins"):
        histogram_init(block, object(), bins=70)


@pytest.mark.parametrize(
    ("primitive", "args"),
    [(histogram_init, ()), (histogram_composite, (object(), object()))],
)
def test_shared_histogram_requires_block(primitive, args):
    with pytest.raises(NotImplementedError, match="requires a block group"):
        primitive(this_warp(), *args)

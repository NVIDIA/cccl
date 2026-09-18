# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Scan requests retain shared result, scratch, and initial-value contracts."""

from dataclasses import replace

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    CxxFunction,
    GroupLoweringTarget,
    LaunchFacts,
    ResultVisibility,
    ScanValueKind,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_warp,
)
from cuda.coop.cutlass import TempStorage, ThreadData
from cuda.coop.cutlass._compiler import _rendering, _state, _storage
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES
from cuda.coop.cutlass._lowering import _scan

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _plan(group=None, **options):
    kwargs = dict(
        group=this_block() if group is None else group,
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        dtype=cutlass.Int32,
        value_kind=ScanValueKind.SCALAR,
        items_per_thread=1,
        mode="exclusive",
        op="sum",
    )
    kwargs.update(options)
    return _scan._make_group_scan_plan(**kwargs).require_supported()


@pytest.mark.parametrize("algorithm", ("raking", "raking_memoize", "warp_scans"))
@pytest.mark.parametrize("array", (False, True))
@pytest.mark.parametrize("mode", ("inclusive", "exclusive"))
def test_block_contracts(algorithm, array, mode):
    plan = _plan(
        algorithm=algorithm,
        mode=mode,
        value_kind=ScanValueKind.ARRAY if array else ScanValueKind.SCALAR,
        items_per_thread=3 if array else 1,
        aggregate=True,
    )
    request = _scan._CubScanRequest(
        _scan._with_block_storage(plan, None), "sum", cutlass.Int32, True
    )
    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.has_aggregate
    assert plan.result.result_items_per_thread == (3 if array else 1)
    assert request.plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert request.plan.temp_storage.exact_layout_required
    assert request.plan.temp_storage.sharing is None
    assert (
        request.plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    )
    assert len(_rendering.bundle_scratch_layout_probes([request])) == 1


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("mode", ("inclusive", "exclusive"))
def test_warp_prefix_contracts(width, mode):
    plan = _plan(
        this_warp().group_by(width),
        mode=mode,
        valid_items=cutlass.Int32(1),
        aggregate=True,
    )
    request = _scan._CubScanRequest(plan, "sum", cutlass.Int32)
    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert _scan._warp_instances(plan) == (64 // width, width)
    assert plan.participation.argument_preconditions[0].minimum == 1
    assert plan.participation.argument_preconditions[0].maximum == width
    assert plan.result.has_aggregate
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
    assert not _rendering.bundle_scratch_layout_probes([request])
    if mode == "exclusive":
        assert isinstance(request.operation.initial_value, CxxFunction)
        assert request.has_initial_value
        assert not request.has_runtime_initial


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
@pytest.mark.parametrize("op", ("sum", "multiplies", "min", "max"))
def test_request_dtype_profile(dtype, op):
    request = _scan._CubScanRequest(
        _plan(dtype=dtype, op=op, initial_value=1), op, dtype
    )
    assert request.value_type is dtype
    assert request.has_runtime_initial


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (True, False))
def test_descriptor_contract(sharing, auto_sync):
    descriptor = TempStorage(8192, alignment=128, sharing=sharing, auto_sync=auto_sync)
    plan = _scan._with_block_storage(_plan(), descriptor)
    request = _scan._CubScanRequest(plan, "sum", cutlass.Int32, True)
    assert plan.temp_storage.ownership is StorageOwnership.CALLER
    assert plan.temp_storage.sharing == sharing
    assert plan.temp_storage.requested_size_in_bytes == 8192
    assert plan.temp_storage.requested_alignment == 128
    assert plan.temp_storage.auto_sync is auto_sync
    expected = SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    assert plan.synchronization.storage_reuse_barrier is expected
    other = _scan._CubScanRequest(
        _scan._with_block_storage(_plan(), TempStorage(auto_sync=not auto_sync)),
        "sum",
        cutlass.Int32,
        True,
    )
    assert request.symbol_name != other.symbol_name
    assert request.scratch_requirement_key == other.scratch_requirement_key


@pytest.mark.parametrize("valid", (0, -1, 9))
def test_prefix_bounds(valid):
    with pytest.raises((ValueError, NotImplementedError), match="valid_items"):
        _plan(this_warp().group_by(8), valid_items=valid)


def test_request_plan_mismatch():
    plan = _plan()
    with pytest.raises(ValueError, match="dtype"):
        _scan._CubScanRequest(plan, "sum", cutlass.Uint32)
    with pytest.raises(ValueError, match="operator"):
        _scan._CubScanRequest(plan, "min", cutlass.Int32)
    with pytest.raises(ValueError, match="method"):
        _scan._CubScanRequest(
            replace(
                plan,
                implementation=replace(
                    plan.implementation,
                    algorithm=replace(
                        plan.implementation.algorithm, method_name="InclusiveSum"
                    ),
                ),
            ),
            "sum",
            cutlass.Int32,
        )


@pytest.mark.parametrize("initial", (np.float32(np.inf), np.float32(np.nan)))
def test_nonfinite_seed(initial):
    with pytest.raises(ValueError, match="finite"):
        _scan._typed_value(initial, cutlass.Float32, name="initial_value", initial=True)


def test_seed_dtype_and_range():
    assert isinstance(
        _scan._typed_value(2, cutlass.Float32, initial=True), cutlass.Float32
    )
    with pytest.raises(TypeError, match="dtype must match"):
        _scan._typed_value(np.float64(2), cutlass.Float32, initial=True)
    with pytest.raises(ValueError, match="not representable"):
        _scan._typed_value(1 << 200, cutlass.Float32, initial=True)


@pytest.mark.parametrize("block", (False, True))
def test_failed_ffi_rolls_back(block, monkeypatch):
    snapshot = object()
    registered, restored, events = [], [], []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: snapshot)
    monkeypatch.setattr(_state, "register_request", registered.append)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)

    def record_storage(descriptor, **kwargs):
        events.append(descriptor)
        return ()

    def fail_ffi(**kwargs):
        raise RuntimeError("scan FFI failed")

    monkeypatch.setattr(
        _storage, "register_deferred_temp_storage_event", record_storage
    )
    monkeypatch.setattr(_scan, "ffi", fail_ffi)
    with pytest.raises(RuntimeError, match="scan FFI failed"):
        _scan.provider_scan(
            group=this_block() if block else this_warp(),
            launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
            value=3,
        )
    assert len(registered) == 1
    assert len(events) == int(block)
    assert restored == [snapshot]


@pytest.mark.parametrize(
    "output", (object(), ThreadData(2), ThreadData(1, dtype=cutlass.Float32))
)
def test_aggregate_shape_and_dtype(output):
    with pytest.raises((TypeError, ValueError), match="aggregate_output"):
        _scan._validate_aggregate_output(output, value_type=cutlass.Int32)


@pytest.mark.parametrize("value", (True, np.bool_(True), 1.5))
def test_prefix_type_error(value):
    with pytest.raises(
        TypeError, match=r"cutlass\.scan valid_items must be an integer"
    ):
        _plan(this_warp(), valid_items=value)

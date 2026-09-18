# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Exchange requests retain layout, auxiliary dtype, and group contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop._core import (
    BlockExchangeMode,
    GroupLoweringTarget,
    LaunchFacts,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_warp,
)
from cuda.coop.cutlass import ThreadData
from cuda.coop.cutlass._compiler import _rendering, _state
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES, INTEGER_VALUE_TYPES
from cuda.coop.cutlass._group_exchange import _normalize_exchange_mode
from cuda.coop.cutlass._lowering import _exchange

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(group=None, *, mode="striped_to_blocked", block=(8, 4, 2), **options):
    mode = BlockExchangeMode(mode)
    kwargs = dict(
        group=this_block() if group is None else group,
        launch=LaunchFacts(exact_block_dim=block),
        dtype=cutlass.Int32,
        items_per_thread=2,
        mode=mode.value,
        rank_dtype=cutlass.Int32 if mode.uses_ranks else None,
        valid_flag_dtype=cutlass.Int32 if mode.uses_valid_flags else None,
    )
    kwargs.update(options)
    plan = _exchange._make_group_exchange_plan(**kwargs).require_supported()
    return _exchange._CubExchangeRequest(
        plan, kwargs["dtype"], kwargs["rank_dtype"], kwargs["valid_flag_dtype"]
    )


@pytest.mark.parametrize("mode", tuple(mode.value for mode in BlockExchangeMode))
def test_block_mode_contract(mode):
    request = _request(mode=mode)
    assert request.plan.target is GroupLoweringTarget.CUB_BLOCK
    assert request.plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert (
        request.plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    )
    assert request.plan.result.visibility is ResultVisibility.PER_MEMBER
    assert request.plan.result.result_items_per_thread == 2
    assert request.operation.uses_ranks == (request.rank_type is not None)
    assert request.operation.uses_valid_flags == (request.valid_flag_type is not None)


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("mode", ("striped_to_blocked", "blocked_to_striped"))
def test_warp_instances(width, mode):
    request = _request(this_warp().group_by(width), mode=mode)
    assert request.plan.target is GroupLoweringTarget.CUB_WARP
    assert _exchange._warp_instances(request) == (64 // width, width)
    assert (
        request.plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
    )
    one = _request(this_warp().group_by(width), mode=mode, block=(32, 1, 1))
    assert request.symbol_name != one.symbol_name
    source = _rendering.render_bundle_source([request])
    assert "__syncthreads" not in source
    assert "__activemask" not in source


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
def test_value_types(dtype):
    request = _request(dtype=dtype)
    assert request.value_type is dtype


@pytest.mark.parametrize(
    "dtype", (cutlass.Int8, cutlass.Int16, cutlass.Int32, cutlass.Int64)
)
def test_signed_ranks(dtype):
    assert _request(mode="scatter_to_blocked", rank_dtype=dtype).rank_type is dtype


@pytest.mark.parametrize("dtype", tuple(INTEGER_VALUE_TYPES))
def test_integer_flags(dtype):
    request = _request(mode="scatter_to_striped_flagged", valid_flag_dtype=dtype)
    assert request.valid_flag_type is dtype


@pytest.mark.parametrize(
    "dtype",
    (cutlass.Uint8, cutlass.Uint16, cutlass.Uint32, cutlass.Uint64, cutlass.Float32),
)
def test_rank_type_rejected(dtype):
    with pytest.raises(TypeError, match="ranks.*signed integer"):
        _request(mode="scatter_to_blocked", rank_dtype=dtype)


def test_float_flag_rejected():
    with pytest.raises(TypeError, match="valid_flags.*integer"):
        _request(mode="scatter_to_striped_flagged", valid_flag_dtype=cutlass.Float32)


@pytest.mark.parametrize(
    "mode", ("scatter_to_striped_guarded", "scatter_to_striped_flagged")
)
def test_time_slicing_restriction(mode):
    with pytest.raises(ValueError, match="warp_time_slicing is not supported"):
        _request(mode=mode, warp_time_slicing=True)


def test_request_plan_mismatch():
    request = _request()
    with pytest.raises(ValueError, match="dtype does not match"):
        _exchange._CubExchangeRequest(request.plan, cutlass.Uint32)
    plan = replace(
        request.plan,
        implementation=replace(
            request.implementation,
            algorithm=replace(
                request.implementation.algorithm, method_name="ScatterToBlocked"
            ),
        ),
    )
    with pytest.raises(ValueError, match="method does not match"):
        _exchange._CubExchangeRequest(plan, cutlass.Int32)


@pytest.mark.parametrize("kind", ("warp", "threads_within_warp"))
def test_warp_scatter_profile(kind):
    with pytest.raises(ValueError, match="mode for .* groups must be one of"):
        _normalize_exchange_mode("scatter_to_striped", group_kind=kind)


@pytest.mark.parametrize("group", (this_block(), this_warp().group_by(8)))
def test_failed_ffi_rollback(group, monkeypatch):
    request = _request(group)
    payload = ThreadData(2, dtype=cutlass.Int32, values=[1, 2], alignment=64)
    snapshot = object()
    registered, restored, allocations = [], [], []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: snapshot)
    monkeypatch.setattr(_state, "register_request", registered.append)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(
        _exchange,
        "llvm",
        SimpleNamespace(PointerType=SimpleNamespace(get=lambda space: object())),
    )

    def allocate(*args):
        allocations.append(args)
        return object()

    def fail_ffi(**kwargs):
        raise RuntimeError("Exchange FFI failed")

    monkeypatch.setattr(_exchange, "_make_rmem_tensor", allocate)
    monkeypatch.setattr(_exchange, "ffi", fail_ffi)
    with pytest.raises(RuntimeError, match="Exchange FFI failed"):
        _exchange.provider_exchange(plan=request.plan, value=payload)
    assert allocations == [(2, cutlass.Int32, 64)]
    assert registered == [request]
    assert restored == [snapshot]
    assert payload.values("exchange") == (1, 2)

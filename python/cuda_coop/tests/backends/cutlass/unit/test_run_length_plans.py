# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Run Length Decode plan, ABI, scratch, and transaction contracts."""

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

cutlass = pytest.importorskip("cutlass")

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop._core import (
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    this_block,
    this_warp,
)
from cuda.coop._core.api._dispatch import _compiler_scope
from cuda.coop.cutlass._compiler import _rendering, _state, _storage, _types
from cuda.coop.cutlass._lowering import _run_length

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(**kwargs):
    options = dict(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
        value_type=cutlass.Int32,
        length_type=cutlass.Uint64,
        runs=2,
        decoded=3,
    )
    options.update(kwargs)
    return _run_length._CubRunLengthRequest(
        _run_length._make_run_length_plan(**options)
    )


@pytest.mark.parametrize("bulk", (False, True))
@pytest.mark.parametrize("dtype", tuple(_types.ALL_PROVIDER_TYPES))
def test_numeric_value_plans(dtype, bulk):
    request = _request(value_type=dtype, bulk=bulk)
    result = request.plan.result.values[0]
    assert result.dtype is (cutlass.Uint32 if bulk else dtype)
    assert result.items_per_member == (1 if bulk else 3)
    assert request.implementation.method_name == ("Into" if bulk else "Window")


@pytest.mark.parametrize("dtype", tuple(_types.INTEGER_VALUE_TYPES))
def test_length_plans(dtype):
    assert _request(length_type=dtype).operation.run_length_dtype is dtype


@pytest.mark.parametrize("dtype", (cutlass.Float32, cutlass.Float64))
def test_float_length_rejected(dtype):
    with pytest.raises(TypeError, match="integer lengths"):
        _request(length_type=dtype)


@pytest.mark.parametrize("offset", (-1, 1 << 64))
def test_static_offset_bounds(offset):
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        _request(offset=offset)


@pytest.mark.parametrize("offset", (True, 1.5, None))
def test_integer_offset_required(offset):
    with pytest.raises(TypeError, match="integer"):
        _request(offset=offset)


@pytest.mark.parametrize("extent", (0, -1, True, 1.5, 1 << 31))
def test_output_extent_validation(extent):
    with pytest.raises((ValueError, TypeError), match="positive|integer|signed 32-bit"):
        _request(decoded=extent)


def test_group_and_dimensions():
    with pytest.raises((ValueError, NotImplementedError), match="this_block"):
        _request(group=this_warp())
    with pytest.raises(ValueError, match="one-dimensional"):
        _request(launch=LaunchFacts(exact_block_dim=(8, 4, 1)))


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_storage_contract(sharing, auto_sync):
    request = _request(
        temp_storage=cutlass_coop.TempStorage(
            alignment=128, sharing=sharing, auto_sync=auto_sync
        )
    )
    assert request.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert request.plan.temp_storage.requested_alignment == 128
    assert request.plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


def test_typed_identity_and_driver_deduplication():
    requests = [
        _request(),
        _request(bulk=True),
        _request(offset=(1 << 64) - 1),
        _request(length_type=cutlass.Int64),
    ]
    assert len({request.symbol_name for request in requests}) == 4
    source = _rendering.render_bundle_source(requests)
    assert source.count("class BlockRunLengthDecodeCoop") == 1
    assert "18446744073709551615ULL" in source
    assert "total > static_cast<wide_t>(capacity) - start" in source
    assert "negative(lengths[i])" in source
    assert "a.has_zero && b.count != 0" in source


def test_result_mismatch_rejected():
    request = _request()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], dtype=cutlass.Float64),),
    )
    with pytest.raises(ValueError, match="result"):
        _run_length._CubRunLengthRequest(replace(request.plan, result=result))


@pytest.mark.parametrize("name", ("run_length_decode", "run_length_decode_into"))
def test_common_signature(name):
    expected, actual = (
        inspect.signature(getattr(coop, name)),
        inspect.signature(getattr(cutlass_coop, name)),
    )
    assert tuple(actual.parameters) == tuple(expected.parameters)
    for name in actual.parameters:
        assert actual.parameters[name].kind is expected.parameters[name].kind
        assert actual.parameters[name].default == expected.parameters[name].default


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("primitive", ("run_length_decode", "run_length_decode_into"))
@pytest.mark.parametrize("invalid", ("run_values", "run_lengths"))
def test_invalid_payload_names_operation(api, primitive, invalid):
    values = cutlass_coop.ThreadData(1, dtype=cutlass.Int32, values=[3])
    lengths = cutlass_coop.ThreadData(1, dtype=cutlass.Uint32, values=[2])
    args = [
        this_block(),
        object() if invalid == "run_values" else values,
        object() if invalid == "run_lengths" else lengths,
    ]
    if primitive == "run_length_decode_into":
        args.append(object())
    with _compiler_scope("cuda.coop.cutlass"):
        with pytest.raises(TypeError, match=rf"\.{primitive} ") as error:
            getattr(api, primitive)(*args, decoded_items_per_thread=2)
    assert invalid in str(error.value)
    assert "ThreadData" in str(error.value)


def test_failed_storage_restores_session(monkeypatch):
    saved, restored = object(), []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(_state, "register_request", lambda request: None)
    monkeypatch.setattr(
        _run_length, "_make_rmem_tensor", lambda *args: SimpleNamespace()
    )

    def fail(*args, **kwargs):
        raise RuntimeError("run length scratch emission failed")

    monkeypatch.setattr(_storage, "register_deferred_temp_storage_event", fail)
    values = cutlass_coop.ThreadData(1, dtype=cutlass.Int32, values=[3])
    lengths = cutlass_coop.ThreadData(1, dtype=cutlass.Uint32, values=[2])
    with pytest.raises(RuntimeError, match="scratch emission failed"):
        _run_length.provider_run_length_decode(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
            values=values,
            lengths=lengths,
            decoded_items_per_thread=2,
            offset=0,
            destination=None,
            bulk=False,
            temp_storage=None,
        )
    assert restored == [saved]
    assert values.values("run_length_decode") == (3,)
    assert lengths.values("run_length_decode") == (2,)

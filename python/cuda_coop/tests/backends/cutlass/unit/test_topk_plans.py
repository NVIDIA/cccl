# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Shared TopK policy, checked count ABI, and storage transactions."""

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
from cuda.coop.cutlass._compiler import _rendering, _state, _storage, _types
from cuda.coop.cutlass._lowering import _topk

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(**kwargs):
    options = dict(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=cutlass.Int32,
        value_type=None,
        items=2,
        selection="min",
        k=17,
        valid_items=None,
    )
    options.update(kwargs)
    return _topk._CubTopKRequest(_topk._make_topk_plan(**options))


@pytest.mark.parametrize("dtype", tuple(_types.ALL_PROVIDER_TYPES))
@pytest.mark.parametrize("selection", ("min", "max"))
def test_numeric_plans(dtype, selection):
    request = _request(key_type=dtype, value_type=cutlass.Float64, selection=selection)
    assert request.implementation.struct_name == "BlockTopKCoop"
    assert request.implementation.method_name == f"{selection}_pairs_full"
    assert tuple(result.dtype for result in request.plan.result.values) == (
        dtype,
        cutlass.Float64,
    )


@pytest.mark.parametrize("name", ("k", "valid_items"))
@pytest.mark.parametrize("value", (-1, 129, 1 << 32))
def test_static_count_bounds(name, value):
    with pytest.raises(ValueError, match=r"in \[0, 128\]"):
        _request(**{name: value})


@pytest.mark.parametrize("name", ("k", "valid_items"))
@pytest.mark.parametrize("value", (True, 1.5))
def test_noninteger_counts(name, value):
    with pytest.raises(TypeError, match="integer"):
        _request(**{name: value})


def test_required_k_cannot_be_omitted():
    with pytest.raises(TypeError, match="integer"):
        _request(k=None)


def test_full_prefix_identity():
    full, explicit = _request(), _request(valid_items=128)
    assert full.symbol_name != explicit.symbol_name
    assert full.implementation.method_name == "min_keys_full"
    assert explicit.implementation.method_name == "min_keys_partial"
    source = _rendering.render_bundle_source([full, explicit])
    assert source.count("class BlockTopKCoop") == 1
    assert source.index("class BlockTopKCoop") < source.index('extern "C"')
    assert "static_cast<int>(k)" in source


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_storage_controls(sharing, auto_sync):
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


def test_group_and_dimensions():
    with pytest.raises((ValueError, NotImplementedError), match="this_block"):
        _request(group=this_warp())
    with pytest.raises(ValueError, match="one-dimensional"):
        _request(launch=LaunchFacts(exact_block_dim=(8, 4, 2)))


def test_result_mismatch():
    request = _request()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], dtype=cutlass.Float64),),
    )
    with pytest.raises(ValueError, match="result dtypes"):
        _topk._CubTopKRequest(replace(request.plan, result=result))


@pytest.mark.parametrize(
    "name", ("topk_min_keys", "topk_max_keys", "topk_min_pairs", "topk_max_pairs")
)
def test_common_signature(name):
    expected = inspect.signature(getattr(coop, name))
    actual = inspect.signature(getattr(cutlass_coop, name))
    assert tuple(actual.parameters) == tuple(expected.parameters)
    for name in actual.parameters:
        assert actual.parameters[name].kind is expected.parameters[name].kind
        assert actual.parameters[name].default == expected.parameters[name].default


def test_failed_storage_restores_session(monkeypatch):
    saved, restored = object(), []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(_state, "register_request", lambda request: None)
    monkeypatch.setattr(_topk, "_make_rmem_tensor", lambda *args: SimpleNamespace())

    def fail(*args, **kwargs):
        raise RuntimeError("TopK scratch emission failed")

    monkeypatch.setattr(_storage, "register_deferred_temp_storage_event", fail)
    keys = cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[3, 1])
    with pytest.raises(RuntimeError, match="scratch emission failed"):
        _topk.provider_topk(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
            keys=keys,
            values=None,
            selection="min",
            k=17,
            valid_items=None,
            temp_storage=None,
        )
    assert restored == [saved]
    assert keys.values("topk") == (3, 1)

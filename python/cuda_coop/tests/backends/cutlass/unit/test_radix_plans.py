# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Radix plan, ABI, storage, and side-output transaction contracts."""

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
from cuda.coop.cutlass import TempStorage, ThreadData, radix_rank
from cuda.coop.cutlass._compiler import _rendering, _state, _storage
from cuda.coop.cutlass._lowering import _radix

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _sort(**kwargs):
    options = dict(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        key_type=cutlass.Int32,
        value_type=None,
        items=3,
        scalar=False,
        begin_bit=0,
        end_bit=32,
        descending=False,
        blocked_to_striped=False,
    )
    options.update(kwargs)
    return _radix._CubRadixRequest(_radix._sort_plan(**options))


def _rank(**kwargs):
    options = dict(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        key_type=cutlass.Int32,
        items=3,
        scalar=False,
        begin_bit=0,
        end_bit=4,
        descending=False,
        prefix_items=None,
    )
    options.update(kwargs)
    return _radix._CubRadixRequest(_radix._rank_plan(**options))


@pytest.mark.parametrize("dtype", tuple(_radix._SORT_KEYS))
@pytest.mark.parametrize("pairs", (False, True))
def test_sort_plans(dtype, pairs):
    request = _sort(
        key_type=dtype, value_type=cutlass.Uint8 if pairs else None, end_bit=32
    )
    assert request.plan.result.values[0].dtype is dtype
    assert len(request.plan.result.values) == 1 + pairs
    assert request.implementation.struct_name == "CudaCoopBlockRadixSort"


@pytest.mark.parametrize("begin,end", ((-1, 4), (4, 4), (7, 3), (0, 33), (1 << 32, 32)))
def test_invalid_static_bits(begin, end):
    with pytest.raises(ValueError, match="bit"):
        _sort(begin_bit=begin, end_bit=end)


@pytest.mark.parametrize(
    "dtype", (cutlass.Int32, cutlass.Uint32, cutlass.Int64, cutlass.Uint64)
)
@pytest.mark.parametrize("descending", (False, True))
def test_rank_ordered_bits(dtype, descending):
    request = _rank(key_type=dtype, descending=descending)
    source = _rendering.render_bundle_source([request])
    assert ("0x80000000u" in source) == (dtype is cutlass.Int32)
    assert ("0x8000000000000000ull" in source) == (dtype is cutlass.Int64)
    assert (
        request.implementation.template_arguments["SMEM_CONFIG"]
        == "cudaSharedMemBankSizeFourByte"
    )


def test_sort_runtime_identity():
    assert (
        _sort(begin_bit=0, end_bit=4).symbol_name
        == _sort(begin_bit=8, end_bit=12).symbol_name
    )
    assert _sort().symbol_name != _sort(descending=True).symbol_name
    assert _sort().symbol_name != _sort(blocked_to_striped=True).symbol_name


def test_prefix_identity_and_initialization():
    request = _rank(prefix_items=1)
    assert request.symbol_name != _rank().symbol_name
    assert "prefix[i] = -1" in _rendering.render_bundle_source([request])


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_storage_policy(sharing, auto_sync):
    request = _sort(
        temp_storage=TempStorage(alignment=128, sharing=sharing, auto_sync=auto_sync)
    )
    assert request.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert request.plan.temp_storage.requested_alignment == 128
    assert request.plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


@pytest.mark.parametrize("factory", (_sort, _rank))
def test_physical_block_required(factory):
    with pytest.raises((ValueError, NotImplementedError), match="physical block"):
        factory(group=this_warp())


def test_result_dtype_validated():
    request = _rank()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], dtype=cutlass.Uint32),),
    )
    with pytest.raises(ValueError, match="result dtypes"):
        _radix._CubRadixRequest(replace(request.plan, result=result))


def test_prefix_alias_rejected_before_snapshot():
    keys = ThreadData(1, dtype=cutlass.Int32, values=[2])
    with pytest.raises(ValueError, match="distinct"):
        radix_rank(this_block(), keys, exclusive_digit_prefix=keys)


def test_failed_ffi_preserves_prefix_and_session(monkeypatch):
    request = _rank(prefix_items=1)
    keys = ThreadData(3, dtype=cutlass.Int32, values=[3, 1, 2])
    prefix = ThreadData(1, values=[-7])
    saved, restored = object(), []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(_state, "register_request", lambda request: None)
    monkeypatch.setattr(_radix, "_typed_item", lambda value, dtype: value)
    monkeypatch.setattr(
        _radix,
        "_make_rmem_tensor",
        lambda *args: SimpleNamespace(iterator=SimpleNamespace(llvm_ptr=0)),
    )
    monkeypatch.setattr(
        _radix,
        "llvm",
        SimpleNamespace(PointerType=SimpleNamespace(get=lambda space: 0)),
    )
    monkeypatch.setattr(
        _storage,
        "register_deferred_temp_storage_event",
        lambda *args, **kwargs: (0, 0, 0),
    )

    def fail(*args):
        raise RuntimeError("injected radix ffi failure")

    monkeypatch.setattr(_radix, "ffi", lambda **kwargs: fail)
    with pytest.raises(RuntimeError, match="injected radix ffi failure"):
        _radix._materialize(
            request, [keys], [(cutlass.Int32, (3, 1, 2))], prefix=prefix
        )
    assert restored == [saved]
    assert prefix.dtype is None
    assert prefix.values("radix_rank") == (-7,)
    assert keys.values("radix_rank") == (3, 1, 2)

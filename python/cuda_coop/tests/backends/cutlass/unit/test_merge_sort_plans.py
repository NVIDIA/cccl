# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Merge Sort contracts include readonly inputs and checked partial tiles."""

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
from cuda.coop._core.api._dispatch import _common_root_operation_scope
from cuda.coop.cutlass import TempStorage, ThreadData
from cuda.coop.cutlass._compiler import _rendering, _state, _storage
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES
from cuda.coop.cutlass._lowering import _merge_sort
from cuda.coop.cutlass._thread_data import _snapshot_readable_payload

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(*, group=None, count=None, pairs=False, storage=None, **options):
    kwargs = dict(
        group=this_block() if group is None else group,
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        key_type=cutlass.Int32,
        value_type=cutlass.Float64 if pairs else None,
        items_per_thread=3,
        descending=False,
        valid_items=count,
        temp_storage=storage,
    )
    kwargs.update(options)
    return _merge_sort._CubMergeSortRequest(_merge_sort._make_merge_sort_plan(**kwargs))


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
@pytest.mark.parametrize("pairs", (False, True))
def test_numeric_results(dtype, pairs):
    request = _request(key_type=dtype, pairs=pairs)
    assert request.plan.result.values[0].dtype is dtype
    assert len(request.plan.result.values) == (2 if pairs else 1)
    assert request.plan.temp_storage.exact_layout_required


@pytest.mark.parametrize("count", (-1, 193, 1 << 32, True, 1.5))
def test_invalid_count(count):
    with pytest.raises((TypeError, ValueError), match="valid_items"):
        _request(count=count)


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
def test_group_storage(width):
    request = _request(group=this_warp().group_by(width), count=0)
    assert request.plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert (
        request.plan.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
    )
    assert request.plan.temp_storage.instances == 64 // width


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_block_storage_controls(sharing, auto_sync):
    storage = TempStorage(alignment=128, sharing=sharing, auto_sync=auto_sync)
    request = _request(storage=storage)
    assert request.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert request.plan.temp_storage.sharing == sharing
    assert request.plan.temp_storage.requested_alignment == 128
    assert request.plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK if auto_sync else SynchronizationScope.NONE
    )


def test_semantic_changes_do_not_share_provider():
    requests = [
        _request(),
        _request(count=0),
        _request(count=1),
        _request(pairs=True),
        _request(descending=True),
    ]
    assert len({request.symbol_name for request in requests}) == len(requests)


def test_partial_definition_order():
    requests = [_request(count=0), _request(group=this_warp(), count=0)]
    source = _rendering.render_bundle_source(requests)
    assert source.index("struct CudaCoopCheckedMergeSort") < source.index(
        "using CudaCoopBlockMergeSort"
    )
    assert source.index("struct CudaCoopCheckedMergeSort") < source.index(
        "using CudaCoopWarpMergeSort"
    )
    assert source == _rendering.render_bundle_source(reversed(requests))


def test_mismatched_result_rejected():
    request = _request()
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], dtype=cutlass.Float64),),
    )
    with pytest.raises(ValueError, match="result dtypes"):
        _merge_sort._CubMergeSortRequest(replace(request.plan, result=result))


class _Readonly:
    def __init__(self, dtype):
        self.items_per_thread = 3
        self.dtype = dtype
        self.alignment = 64
        self._items = (np.int32(3), np.int32(1), np.int32(2))

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        return self._items[index]


@pytest.mark.parametrize("dtype", (np.int32, None))
@pytest.mark.parametrize("primitive", ("merge_sort_keys", "merge_sort_pairs"))
def test_readonly_snapshot(dtype, primitive):
    source = _Readonly(dtype)
    with _common_root_operation_scope(primitive):
        result = _snapshot_readable_payload(source, name="keys", primitive=primitive)
    assert result.items_per_thread == 3
    assert result.dtype is dtype
    assert result.alignment == 64
    result[0] = 17
    assert list(source) == [3, 1, 2]


def test_readonly_mixed_dtype_rejected():
    source = _Readonly(None)
    source._items = (np.int32(1), np.float32(2), np.int32(3))
    with pytest.raises(TypeError, match="common dtype"):
        _snapshot_readable_payload(source, name="keys", primitive="merge_sort_keys")


def test_failed_storage_emission_restores_session(monkeypatch):
    saved, restored, requests = object(), [], []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "register_request", requests.append)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(
        _merge_sort, "_make_rmem_tensor", lambda *args: SimpleNamespace()
    )

    def fail(*args, **kwargs):
        raise RuntimeError("scratch emission failed")

    monkeypatch.setattr(_storage, "register_deferred_temp_storage_event", fail)
    keys = ThreadData(2, dtype=cutlass.Int32, values=[3, 1])
    with pytest.raises(RuntimeError, match="scratch emission failed"):
        _merge_sort.provider_merge_sort(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
            keys=keys,
            values=None,
            descending=False,
            valid_items=None,
            oob_default=None,
            temp_storage=None,
        )
    assert len(requests) == 1
    assert restored == [saved]
    assert keys.values("merge_sort") == (3, 1)

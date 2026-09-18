# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace
from importlib import import_module

import pytest

pytest.importorskip("cutlass")
_storage = import_module("cuda.coop.cutlass._compiler._storage")
_state = import_module("cuda.coop.cutlass._compiler._state")
_types = import_module("cuda.coop.cutlass._compiler._types")
TempStorage = import_module("cuda.coop.cutlass._temp_storage").TempStorage
ir = import_module("cutlass._mlir.ir")

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _event(storage, key, kernel):
    return _types.DeferredTempStorageEvent(
        kernel_op=kernel,
        kernel_name="example",
        temp_storage=storage,
        primitive_name="load",
        requirement_key=key,
        sharing=storage.sharing,
        auto_sync=storage.auto_sync,
        capacity_size_in_bytes=storage.capacity_size_in_bytes,
        capacity_alignment=storage.alignment,
        smem_addr_placeholder=object(),
        size_placeholder=object(),
        location="example.py:12",
    )


@pytest.mark.parametrize("capacity", [None, 128])
@pytest.mark.parametrize("requested", [None, 1, 8, 16, 32, 64])
def test_shared_uses_one_maximum_slot(capacity, requested):
    storage = TempStorage(capacity, alignment=requested)
    kernel = object()
    events = [_event(storage, key, kernel) for key in ("small", "large")]
    layouts = {
        "small": _types.ScratchLayout(24, 8),
        "large": _types.ScratchLayout(64, 16),
    }
    (plan,) = _storage.plan_deferred_temp_storage_events(events, layouts)
    expected_size = capacity or 64
    expected_alignment = max(requested or 1, 16)
    assert plan.temp_storage is storage
    assert plan.kernel_op is kernel
    assert plan.size_in_bytes == expected_size
    assert plan.alignment == expected_alignment
    assert [binding.byte_offset_in_bytes for binding in plan.bindings] == [0, 0]
    assert [binding.size_in_bytes for binding in plan.bindings] == [
        expected_size,
        expected_size,
    ]
    assert [binding.alignment for binding in plan.bindings] == [
        expected_alignment,
        expected_alignment,
    ]


@pytest.mark.parametrize("capacity", [None, 48, 128])
@pytest.mark.parametrize("requested", [None, 1, 8, 16, 32, 64])
@pytest.mark.parametrize("auto_sync", [None, True, False])
def test_exclusive_slices(capacity, requested, auto_sync):
    storage = TempStorage(
        capacity, alignment=requested, sharing="exclusive", auto_sync=auto_sync
    )
    kernel = object()
    events = [_event(storage, key, kernel) for key in ("first", "second")]
    layouts = {
        "first": _types.ScratchLayout(24, 8),
        "second": _types.ScratchLayout(16, 16),
    }
    (plan,) = _storage.plan_deferred_temp_storage_events(events, layouts)
    assert plan.size_in_bytes == (capacity or 48)
    assert plan.alignment == max(requested or 1, 16)
    assert [binding.byte_offset_in_bytes for binding in plan.bindings] == [0, 32]
    assert [binding.size_in_bytes for binding in plan.bindings] == [24, 16]
    assert [binding.alignment for binding in plan.bindings] == [8, 16]
    assert all(
        binding.event.auto_sync is (True if auto_sync is None else auto_sync)
        for binding in plan.bindings
    )


@pytest.mark.parametrize("sharing,capacity", [("shared", 23), ("exclusive", 47)])
def test_undersized_capacity(sharing, capacity):
    storage = TempStorage(capacity, sharing=sharing)
    kernel = object()
    events = [_event(storage, key, kernel) for key in ("first", "second")]
    layouts = {
        "first": _types.ScratchLayout(24, 8),
        "second": _types.ScratchLayout(16, 16),
    }
    with pytest.raises(_storage.DSLRuntimeError, match="capacity is smaller"):
        _storage.plan_deferred_temp_storage_events(events, layouts)


def test_kernel_and_storage_isolation():
    first, second = TempStorage(), TempStorage(sharing="exclusive")
    kernel_a, kernel_b = object(), object()
    events = [
        _event(first, "small", kernel_a),
        _event(second, "small", kernel_a),
        _event(first, "large", kernel_b),
        _event(second, "small", kernel_a),
    ]
    layouts = {
        "small": _types.ScratchLayout(16, 8),
        "large": _types.ScratchLayout(64, 16),
    }
    plans = _storage.plan_deferred_temp_storage_events(events, layouts)
    assert len(plans) == 3
    assert [plan.size_in_bytes for plan in plans] == [16, 32, 64]
    assert [len(plan.bindings) for plan in plans] == [1, 2, 1]
    assert [plan.kernel_op for plan in plans] == [kernel_a, kernel_a, kernel_b]


@pytest.mark.parametrize(
    "changed",
    [
        {"sharing": "exclusive"},
        {"auto_sync": False},
        {"capacity_size_in_bytes": 64},
        {"capacity_alignment": 32},
    ],
)
def test_configuration_change_rejected(changed):
    storage = TempStorage()
    first = _event(storage, "scratch", object())
    second = replace(first, **changed)
    with pytest.raises(_storage.DSLRuntimeError, match="configuration changed"):
        _storage.plan_deferred_temp_storage_events(
            [first, second], {"scratch": _types.ScratchLayout(16, 8)}
        )


def test_missing_layout_is_diagnostic():
    event = _event(TempStorage(), "missing", object())
    with pytest.raises(_storage.DSLRuntimeError, match=r"No exact C\+\+ scratch"):
        _storage.plan_deferred_temp_storage_events([event], {})


def test_empty_plan():
    assert _storage.plan_deferred_temp_storage_events([], {}) == ()
    assert _storage.materialize_deferred_temp_storage_plans(()) is None


def test_event_rollback():
    session = _state.BundleSession()
    first = _event(TempStorage(), "first", object())
    second = _event(TempStorage(), "second", object())
    session.add_deferred_temp_storage_event(first)
    snapshot = session.snapshot()
    session.add_deferred_temp_storage_event(second)
    assert session.deferred_temp_storage_event_list() == [first, second]
    assert not session.is_empty()
    session.restore(snapshot)
    assert session.deferred_temp_storage_event_list() == [first]
    returned = session.deferred_temp_storage_event_list()
    returned.clear()
    assert session.deferred_temp_storage_event_list() == [first]


def test_registration_uses_fresh_operands(monkeypatch):
    session = _state.BundleSession()
    storage = TempStorage(128, auto_sync=False)
    kernel = object()
    monkeypatch.setattr(_storage, "_active_cuda_kernel_op", lambda: kernel)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            for _ in range(2):
                args = _storage.register_deferred_temp_storage_event(
                    storage,
                    primitive_name="load",
                    requirement_key="scratch",
                    active_session_getter=lambda: session,
                )
                assert len(args) == 3
        first, second = session.deferred_temp_storage_event_list()
        assert first.smem_addr_placeholder != second.smem_addr_placeholder
        assert first.size_placeholder != second.size_placeholder
        assert first.kernel_op is kernel
        assert first.temp_storage is storage
        assert first.capacity_size_in_bytes == 128
        assert first.auto_sync is False


def test_requirement_key_must_be_hashable():
    with pytest.raises(TypeError, match="hashable"):
        _storage.register_deferred_temp_storage_event(
            TempStorage(), primitive_name="load", requirement_key=[]
        )

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test scratch planning from traced calls and exact C++ layouts.

Synthetic events stand in for traced collective calls. The tests supply size
and alignment directly, then check shared slots, exclusive slices, and
isolation by kernel and descriptor. Other tests cover session rollback, fresh
placeholder operands, and kernel discovery.
"""

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
    """Make one traced-call event without generating a provider or GPU kernel.

    Opaque objects stand in for the kernel and the operands that deferred
    allocation will replace. Each test passes layouts to the planner by
    requirement key, so placement checks do not need C++ compilation.
    """

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


def _region_op(name, attributes=None):
    """Build one region and block for a synthetic MLIR nesting test."""

    operation = ir.Operation.create(name, attributes=attributes, regions=1)
    operation.regions[0].blocks.append()
    return operation


@pytest.mark.parametrize("capacity", [None, 128])
@pytest.mark.parametrize("requested", [None, 1, 8, 16, 32, 64])
def test_shared_uses_one_maximum_slot(capacity, requested):
    """Share one slot large enough for the largest collective requirement.

    Both calls start at offset zero. Provider alignment sets a lower bound on
    the allocation; caller capacity can reserve more bytes than the maximum
    requirement without creating another slot.
    """

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
    """Place each call in its own aligned slice of the same allocation.

    The first requirement occupies 24 bytes. The second needs alignment 16,
    so it starts at byte 32 and brings the minimum capacity to 48 bytes. The
    allocation may have stronger alignment without changing either slice's
    native size and alignment.
    """

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
    assert [binding.byte_offset_in_bytes for binding in plan.bindings] == [
        0,
        32,
    ]
    assert [binding.size_in_bytes for binding in plan.bindings] == [24, 16]
    assert [binding.alignment for binding in plan.bindings] == [8, 16]
    assert all(
        binding.event.auto_sync is (False if auto_sync is None else auto_sync)
        for binding in plan.bindings
    )


@pytest.mark.parametrize(
    "sharing,capacity", [("shared", 23), ("exclusive", 47)]
)
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
        {"auto_sync": True},
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
    with pytest.raises(
        _storage.DSLRuntimeError, match=r"No exact C\+\+ scratch"
    ):
        _storage.plan_deferred_temp_storage_events([event], {})


def test_empty_plan():
    assert _storage.plan_deferred_temp_storage_events([], {}) == ()
    assert _storage.materialize_deferred_temp_storage_plans(()) is None


def test_event_rollback():
    """Restore the call list when a tracing attempt is rolled back.

    A snapshot keeps the first call while discarding the later call. Reading
    the event list must return a copy, so callers cannot change session state
    by editing the returned list.
    """

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


@pytest.mark.parametrize("kernel_name", ("cuda.kernel", "lir.func"))
@pytest.mark.parametrize(
    "nested", (False, True), ids=("entry", "nested-region")
)
def test_registration_uses_fresh_operands(kernel_name, nested):
    """Give each traced call fresh operands and its enclosing kernel.

    One descriptor is reused in two kernels on purpose. Each call needs its
    own address and size placeholders, and planning groups the calls by
    kernel. A call in a nested region must resolve to the same kernel as a
    call in the entry block. Unregistered MLIR operations test the traversal
    without compiling a kernel.
    """

    session = _state.BundleSession()
    storage = TempStorage(128, auto_sync=False)
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_module = _region_op("gpu.module")
        kernels = []
        for name in ("first", "second"):
            attributes = {"sym_name": ir.StringAttr.get(name)}
            if kernel_name == "lir.func":
                attributes["gpu.kernel"] = ir.UnitAttr.get()
            with ir.InsertionPoint(gpu_module.regions[0].blocks[0]):
                kernel = _region_op(kernel_name, attributes)
            kernels.append(kernel)
            block = kernel.regions[0].blocks[0]
            if nested:
                with ir.InsertionPoint(block):
                    region = _region_op("test.region")
                block = region.regions[0].blocks[0]
            with ir.InsertionPoint(block):
                for _ in range(2):
                    args = _storage.register_deferred_temp_storage_event(
                        storage,
                        primitive_name="load",
                        requirement_key="scratch",
                        active_session_getter=lambda: session,
                    )
                    assert len(args) == 3
        events = session.deferred_temp_storage_event_list()
        assert len(events) == 4
        assert len({event.smem_addr_placeholder for event in events}) == 4
        assert len({event.size_placeholder for event in events}) == 4
        assert [event.kernel_op for event in events] == [
            kernels[0],
            kernels[0],
            kernels[1],
            kernels[1],
        ]
        for event in events:
            assert event.temp_storage is storage
            assert event.capacity_size_in_bytes == 128
            assert event.auto_sync is False
        plans = _storage.plan_deferred_temp_storage_events(
            events, {"scratch": _types.ScratchLayout(64, 16)}
        )
        assert [plan.kernel_op for plan in plans] == kernels
        assert [len(plan.bindings) for plan in plans] == [2, 2]


@pytest.mark.parametrize(
    "parent_name,function_name,attribute_names",
    [
        pytest.param("gpu.module", "lir.func", (), id="device-function"),
        pytest.param(
            "gpu.module", "lir.func", ("cu_attrs",), id="launch-attrs-only"
        ),
        pytest.param(
            "builtin.module", "lir.func", ("gpu.kernel",), id="wrong-parent"
        ),
        pytest.param(
            "gpu.module", "func.func", ("gpu.kernel",), id="other-function"
        ),
    ],
)
def test_kernel_discovery_rejects_non_kernel_functions(
    parent_name, function_name, attribute_names
):
    """Reject functions that resemble a kernel but are not one.

    Only ``cuda.kernel``, or an ``lir.func`` that has ``gpu.kernel`` and sits
    directly in ``gpu.module``, owns scratch. Launch attributes alone must not
    make a device helper own kernel scratch.
    """

    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            parent = _region_op(parent_name)
        attributes = {
            name: ir.DictAttr.get({})
            if name == "cu_attrs"
            else ir.UnitAttr.get()
            for name in attribute_names
        }
        with ir.InsertionPoint(parent.regions[0].blocks[0]):
            function = _region_op(function_name, attributes)
        with (
            ir.InsertionPoint(function.regions[0].blocks[0]),
            pytest.raises(
                _storage.DSLRuntimeError, match="enclosing CUDA kernel"
            ),
        ):
            _storage._active_cuda_kernel_op()


def test_kernel_discovery_requires_an_active_trace():
    with (
        ir.Context(),
        ir.Location.unknown(),
        pytest.raises(
            _storage.DSLRuntimeError, match="active CuTe kernel trace"
        ),
    ):
        _storage._active_cuda_kernel_op()


def test_requirement_key_must_be_hashable():
    with pytest.raises(TypeError, match="hashable"):
        _storage.register_deferred_temp_storage_event(
            TempStorage(), primitive_name="load", requirement_key=[]
        )

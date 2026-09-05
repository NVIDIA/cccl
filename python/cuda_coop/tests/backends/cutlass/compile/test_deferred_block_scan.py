# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

pytest.importorskip("cutlass.cute.ffi")
cutlass_common = pytest.importorskip("cutlass.base_dsl.common")
cutlass_typing = pytest.importorskip("cutlass.base_dsl.typing")
coop = pytest.importorskip("cuda.coop.cutlass")
coop_core = pytest.importorskip("cuda.coop._core")
scan_frontend = pytest.importorskip("cuda.coop.cutlass._group_scan")
scan_provider = pytest.importorskip("cuda.coop.cutlass._dsl._cub_scan_provider")
provider_support = pytest.importorskip("cuda.coop.cutlass._dsl._provider")
provider_bundle = pytest.importorskip("cuda.coop.cutlass._dsl._provider_bundle")
single_phase = pytest.importorskip("cuda.coop.cutlass._dsl._single_phase")

DSLRuntimeError = cutlass_common.DSLRuntimeError
Float32 = cutlass_typing.Float32
Float64 = cutlass_typing.Float64
LaunchFacts = coop_core.LaunchFacts
ScanValueKind = coop_core.ScanValueKind


def _scan_plan(
    *,
    value_type=Float32,
    value_kind=ScanValueKind.ARRAY,
    items_per_thread: int = 4,
    group=None,
    aggregate: bool = False,
):
    if group is None:
        group = coop.this_block()
    return scan_frontend._make_group_scan_plan(
        group=group,
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        dtype=value_type,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        mode="exclusive",
        op="sum",
        aggregate=aggregate,
        source="scoped_block",
    ).require_supported()


def _scan_request(
    *,
    value_type=Float32,
    value_kind=ScanValueKind.ARRAY,
    items_per_thread: int = 4,
    external_scratch: bool,
):
    plan = _scan_plan(
        value_type=value_type,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
    )
    if external_scratch:
        plan = scan_provider._with_caller_owned_scan_storage(plan)
    return scan_provider._CubScanRequest(
        plan=plan,
        op="sum",
        value_type=value_type,
        external_scratch=external_scratch,
    )


def test_root_temp_storage_alias_and_inference_semantics():
    assert coop.TempStorage is coop._block.TempStorage
    assert coop.TempStorage is not coop._warp.TempStorage
    assert coop.TempStorage.__module__ == "cuda.coop.cutlass._temp_storage"

    explicit = coop.TempStorage(size_in_bytes=256)
    assert not explicit.is_deferred
    assert explicit.auto_sync

    inferred = coop.TempStorage()
    assert inferred.is_deferred
    assert inferred.capacity_size_in_bytes is None
    assert inferred.alignment is None
    assert inferred.auto_sync

    inferred_manual = coop.TempStorage(auto_sync=False)
    assert inferred_manual.is_deferred
    assert not inferred_manual.auto_sync

    inferred_automatic = coop.TempStorage(auto_sync=True)
    assert inferred_automatic.auto_sync

    exclusive = coop.TempStorage(sharing="exclusive")
    assert exclusive.sharing == "exclusive"
    assert not exclusive.auto_sync

    with pytest.raises(ValueError, match="sharing='exclusive'.*auto_sync=True"):
        coop.TempStorage(sharing="exclusive", auto_sync=True)


def test_root_temp_storage_sync_uses_the_current_block(monkeypatch):
    from cuda.coop.cutlass import _thread_group

    calls = []

    class _Block:
        def sync(self):
            calls.append("sync")

    monkeypatch.setattr(_thread_group, "this_block", _Block)

    coop.TempStorage().sync()

    assert calls == ["sync"]


def test_static_and_external_scan_requests_have_distinct_abis_and_probe():
    static_request = _scan_request(external_scratch=False)
    external_request = _scan_request(external_scratch=True)

    assert static_request != external_request
    assert hash(static_request) != hash(external_request)
    assert external_request.symbol_name == (
        f"{static_request.symbol_name}_external_scratch"
    )

    static_source = "\n".join(scan_provider._render_cub_scan(static_request))
    external_source = "\n".join(scan_provider._render_cub_scan(external_request))

    assert "__shared__ typename implementation_type::TempStorage storage;" in (
        static_source
    )
    assert "cuda_coop_cutlass_block_sync();" in static_source
    assert "unsigned int temp_storage_smem_addr" in external_source
    assert "int temp_storage_bytes" in external_source
    assert "int temp_storage_auto_sync" in external_source
    assert "cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr)" in external_source
    assert "typename implementation_type::TempStorage*>(temp_storage_ptr)" in (
        external_source
    )
    assert "__shared__ typename implementation_type::TempStorage" not in (
        external_source
    )
    assert "if (temp_storage_auto_sync != 0)" in external_source
    assert "cuda_coop_cutlass_block_sync();" in external_source

    assert scan_provider._cub_scan_scratch_layout_probe(static_request) is None
    probe = scan_provider._cub_scan_scratch_layout_probe(external_request)
    assert probe is not None
    assert probe.requirement_key == external_request.scratch_requirement_key
    assert probe.size_expression == f"sizeof({external_request.scratch_cpp_type})"
    assert probe.alignment_expression == (
        f"alignof({external_request.scratch_cpp_type})"
    )
    assert external_request.scratch_cpp_type == (
        "typename ::cub::BlockScan<float, 64, "
        "::cub::BLOCK_SCAN_RAKING, 1, 1>::TempStorage"
    )


def test_scan_layout_key_tracks_cub_class_not_operand_form():
    array_request = _scan_request(
        value_kind=ScanValueKind.ARRAY,
        items_per_thread=4,
        external_scratch=True,
    )
    scalar_request = _scan_request(
        value_kind=ScanValueKind.SCALAR,
        items_per_thread=1,
        external_scratch=True,
    )
    double_request = _scan_request(
        value_type=Float64,
        external_scratch=True,
    )

    assert array_request.scratch_requirement_key == (
        scalar_request.scratch_requirement_key
    )
    assert array_request.scratch_cpp_type == scalar_request.scratch_cpp_type
    assert (
        array_request.scratch_requirement_key != double_request.scratch_requirement_key
    )

    with pytest.raises(ValueError, match="storage ownership"):
        scan_provider._CubScanRequest(
            plan=_scan_plan(),
            op="sum",
            value_type=Float32,
            external_scratch=True,
        )

    warp_plan = _scan_plan(
        value_kind=ScanValueKind.SCALAR,
        items_per_thread=1,
        group=coop.this_warp(),
    )
    warp_plan = scan_provider._with_caller_owned_scan_storage(warp_plan)
    with pytest.raises(ValueError, match="block-scoped only"):
        scan_provider._CubScanRequest(
            plan=warp_plan,
            op="sum",
            value_type=Float32,
            external_scratch=True,
        )


@dataclass(eq=False)
class _FakeKernel:
    name: str


class _Placeholder:
    def __init__(self):
        self.replacements = []

    def replace_all_uses_with(self, replacement):
        self.replacements.append(replacement)


def _event(*, kernel, storage, requirement_key, primitive_name="scan"):
    return provider_support.DeferredTempStorageEvent(
        kernel_op=kernel,
        kernel_name=kernel.name,
        temp_storage=storage,
        primitive_name=primitive_name,
        requirement_key=requirement_key,
        sharing=storage.sharing,
        auto_sync=storage.auto_sync,
        capacity_size_in_bytes=storage.capacity_size_in_bytes,
        capacity_alignment=storage.alignment,
        smem_addr_placeholder=_Placeholder(),
        size_placeholder=_Placeholder(),
        location=f"test:{kernel.name}:scan",
    )


def test_scan_planner_strengthens_layout_and_isolates_storage(monkeypatch):
    first_kernel = _FakeKernel("same_name")
    second_kernel = _FakeKernel("same_name")
    shared_storage = coop.TempStorage()
    other_storage = coop.TempStorage()
    events = [
        _event(
            kernel=first_kernel,
            storage=shared_storage,
            requirement_key="f32",
        ),
        _event(
            kernel=first_kernel,
            storage=shared_storage,
            requirement_key="f64",
        ),
        _event(
            kernel=first_kernel,
            storage=other_storage,
            requirement_key="f32",
        ),
        _event(
            kernel=second_kernel,
            storage=shared_storage,
            requirement_key="f32",
        ),
    ]
    layouts = {
        "f32": provider_support.ScratchLayout(288, 16),
        "f64": provider_support.ScratchLayout(544, 16),
    }

    plans = provider_support.plan_deferred_temp_storage_events(events, layouts)

    assert len(plans) == 3
    strengthened = next(
        plan
        for plan in plans
        if plan.kernel_op is first_kernel and plan.temp_storage is shared_storage
    )
    assert strengthened.size_in_bytes == 544
    assert strengthened.alignment == 16
    assert [binding.size_in_bytes for binding in strengthened.bindings] == [544, 544]
    assert [binding.byte_offset_in_bytes for binding in strengthened.bindings] == [
        0,
        0,
    ]
    assert any(
        plan.kernel_op is first_kernel and plan.temp_storage is other_storage
        for plan in plans
    )
    assert any(
        plan.kernel_op is second_kernel and plan.temp_storage is shared_storage
        for plan in plans
    )


def test_shared_scan_and_exchange_alias_the_strongest_layout(monkeypatch):
    kernel = _FakeKernel("composed")
    storage = coop.TempStorage()
    plans = provider_support.plan_deferred_temp_storage_events(
        [
            _event(kernel=kernel, storage=storage, requirement_key="scan"),
            _event(
                kernel=kernel,
                storage=storage,
                requirement_key="exchange",
                primitive_name="exchange",
            ),
        ],
        {
            "scan": provider_support.ScratchLayout(288, 16),
            "exchange": provider_support.ScratchLayout(1024, 16),
        },
    )

    assert len(plans) == 1
    assert plans[0].size_in_bytes == 1024
    assert plans[0].alignment == 16
    assert [binding.byte_offset_in_bytes for binding in plans[0].bindings] == [
        0,
        0,
    ]
    assert [binding.size_in_bytes for binding in plans[0].bindings] == [1024, 1024]


def test_exclusive_storage_assigns_each_call_an_aligned_slice(monkeypatch):
    kernel = _FakeKernel("exclusive")
    storage = coop.TempStorage(sharing="exclusive")
    plans = provider_support.plan_deferred_temp_storage_events(
        [
            _event(kernel=kernel, storage=storage, requirement_key="scan-f32"),
            _event(kernel=kernel, storage=storage, requirement_key="scan-f64"),
            _event(
                kernel=kernel,
                storage=storage,
                requirement_key="exchange",
                primitive_name="exchange",
            ),
        ],
        {
            "scan-f32": provider_support.ScratchLayout(288, 16),
            "scan-f64": provider_support.ScratchLayout(544, 16),
            "exchange": provider_support.ScratchLayout(1024, 16),
        },
    )

    assert len(plans) == 1
    assert plans[0].size_in_bytes == 1856
    assert plans[0].alignment == 16
    assert [binding.byte_offset_in_bytes for binding in plans[0].bindings] == [
        0,
        288,
        832,
    ]
    assert [binding.size_in_bytes for binding in plans[0].bindings] == [
        288,
        544,
        1024,
    ]


def test_deferred_scan_storage_routes_root_scoped_and_warp(monkeypatch):
    root_storage = coop.TempStorage()

    assert (
        scan_provider._deferred_temp_storage_for_scan(
            group=coop.this_block(),
            source="cutlass_root",
            explicit_temp_storage=root_storage,
        )
        is root_storage
    )

    context = single_phase.SinglePhaseContext(
        thread_data=None,
        temp_storage=root_storage,
    )
    with single_phase.activate_single_phase_context(context):
        assert (
            scan_provider._deferred_temp_storage_for_scan(
                group=coop.this_block(),
                source="scoped_block",
                explicit_temp_storage=None,
            )
            is root_storage
        )
        with pytest.raises(ValueError, match="two TempStorage objects"):
            scan_provider._deferred_temp_storage_for_scan(
                group=coop.this_block(),
                source="scoped_block",
                explicit_temp_storage=coop.TempStorage(),
            )

    with pytest.raises(ValueError, match="only for block groups"):
        scan_provider._deferred_temp_storage_for_scan(
            group=coop.this_warp(),
            source="cutlass_root",
            explicit_temp_storage=root_storage,
        )
    with pytest.raises(ValueError, match="public root or scoped block"):
        scan_provider._deferred_temp_storage_for_scan(
            group=coop.this_block(),
            source="internal_test",
            explicit_temp_storage=root_storage,
        )

    explicit_storage = coop.TempStorage(size_in_bytes=256)
    assert (
        scan_provider._deferred_temp_storage_for_scan(
            group=coop.this_block(),
            source="cutlass_root",
            explicit_temp_storage=explicit_storage,
        )
        is None
    )


def test_root_and_scoped_frontends_preserve_storage_route(
    monkeypatch, set_cutlass_launch_facts
):
    set_cutlass_launch_facts(64)
    storage = coop.TempStorage()
    calls = []

    def capture_provider_scan(**kwargs):
        calls.append((kwargs, single_phase.get_active_single_phase_context()))
        return "scanned"

    monkeypatch.setattr(scan_provider, "provider_scan", capture_provider_scan)

    root_result = coop.scan(
        coop.this_block(),
        object(),
        temp_storage=storage,
    )
    scoped_result = coop._block.exclusive_sum(
        object(),
        temp_storage=storage,
        launch_metadata={"threads_per_block": 64},
    )

    assert root_result == scoped_result == "scanned"
    root_call, root_context = calls[0]
    assert root_call["source"] == "cutlass_root"
    assert root_call["temp_storage"] is storage
    assert root_context is None

    scoped_call, scoped_context = calls[1]
    assert scoped_call["source"] == "scoped_block"
    assert scoped_call["temp_storage"] is None
    assert scoped_context is not None
    assert scoped_context.temp_storage is storage


@pytest.mark.parametrize(
    "primitive_name",
    (
        "exclusive_sum",
        "exclusive_scan",
        "inclusive_sum",
        "inclusive_scan",
        "scan",
    ),
)
def test_all_scoped_block_scan_adapters_accept_keyword_only_storage(
    monkeypatch,
    primitive_name,
):
    storage = coop.TempStorage()
    calls = []

    def capture_provider_scan(**kwargs):
        calls.append((kwargs, single_phase.get_active_single_phase_context()))
        return "scanned"

    monkeypatch.setattr(scan_provider, "provider_scan", capture_provider_scan)
    primitive = getattr(coop._block, primitive_name)

    assert inspect.signature(primitive).parameters["temp_storage"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        primitive(
            object(),
            temp_storage=storage,
            launch_metadata={"threads_per_block": 64},
        )
        == "scanned"
    )
    call, context = calls[-1]
    assert call["source"] == "scoped_block"
    assert call["temp_storage"] is None
    assert context is not None
    assert context.temp_storage is storage

    with pytest.raises(TypeError, match="expects one positional value"):
        primitive(
            object(),
            storage,
            launch_metadata={"threads_per_block": 64},
        )


def test_public_warp_and_unregistered_provider_rejections(
    monkeypatch, set_cutlass_launch_facts
):
    set_cutlass_launch_facts(32)
    storage = coop.TempStorage()

    with pytest.raises(ValueError, match="supported only for block groups"):
        coop.scan(
            coop.this_warp(),
            object(),
            temp_storage=storage,
        )

    with pytest.raises(
        NotImplementedError,
        match=(
            "deferred planning is currently limited to.*"
            "block Load, Store, Exchange, Scan, AdjacentDifference, "
            "Discontinuity, RadixSort, and MergeSort"
        ),
    ):
        coop._warp.exclusive_sum(
            object(),
            temp_storage=coop._warp.TempStorage(),
        )

    with pytest.raises(
        NotImplementedError,
        match=(
            "deferred planning is currently limited to.*"
            "block Load, Store, Exchange, Scan, AdjacentDifference, "
            "Discontinuity, RadixSort, and MergeSort"
        ),
    ):
        coop._block.reduce(object(), temp_storage=storage)


def test_deferred_scan_registration_requires_active_kernel_trace(monkeypatch):
    storage = coop.TempStorage()

    with pytest.raises(DSLRuntimeError, match="active CuTe kernel trace"):
        provider_support.register_deferred_temp_storage_event(
            storage,
            primitive_name="scan",
            requirement_key=("scan", "f32"),
            active_session_getter=provider_support.BundleSession,
        )


def test_scan_materialization_restores_session_after_ffi_failure(monkeypatch):
    storage = coop.TempStorage()
    plan = scan_provider._with_caller_owned_scan_storage(
        _scan_plan(
            value_kind=ScanValueKind.SCALAR,
            items_per_thread=1,
        )
    )
    snapshot = object()
    restored = []
    registrations = []

    monkeypatch.setattr(
        provider_support,
        "snapshot_active_session_state",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        provider_support,
        "restore_active_session_state",
        restored.append,
    )
    monkeypatch.setattr(
        provider_support,
        "register_request",
        lambda request: registrations.append(("request", request)),
    )
    monkeypatch.setattr(
        provider_support,
        "register_deferred_temp_storage_event",
        lambda *args, **kwargs: (
            registrations.append(("event", args, kwargs))
            or (object(), object(), object())
        ),
    )

    def failing_ffi(**_kwargs):
        def invoke(*_args):
            raise RuntimeError("forced FFI failure")

        return invoke

    monkeypatch.setattr(scan_provider, "ffi", failing_ffi)

    with pytest.raises(RuntimeError, match="forced FFI failure"):
        scan_provider._materialize_scan(
            plan=plan,
            value=object(),
            values=(object(),),
            value_type=Float32,
            op="sum",
            initial_value=None,
            aggregate_output=None,
            valid_items=None,
            deferred_temp_storage=storage,
        )

    assert [entry[0] for entry in registrations] == ["request", "event"]
    assert restored == [snapshot]


def test_scan_array_ffi_uses_explicit_llvm_output_pointers(monkeypatch):
    plan = _scan_plan(
        value_kind=ScanValueKind.ARRAY,
        items_per_thread=2,
        aggregate=True,
    )
    tensors = []
    ffi_calls = []

    class FakeTensor:
        def __init__(self):
            class Iterator:
                pass

            self.iterator = Iterator()
            self.iterator.llvm_ptr = object()

        def __getitem__(self, index):
            return Float32(index)

    def make_tensor(*_args):
        tensor = FakeTensor()
        tensors.append(tensor)
        return tensor

    def capture_ffi(**_kwargs):
        def invoke(*args):
            ffi_calls.append(args)

        return invoke

    monkeypatch.setattr(scan_provider._cute, "make_rmem_tensor", make_tensor)
    monkeypatch.setattr(provider_support, "register_request", lambda _request: None)
    monkeypatch.setattr(scan_provider, "ffi", capture_ffi)
    monkeypatch.setattr(scan_provider.llvm.PointerType, "get", lambda *_args: object())

    aggregate = coop.ThreadData(1, dtype=Float32)
    value = coop.ThreadData.from_values(Float32(3), Float32(5), dtype=Float32)
    scan_provider._materialize_scan(
        plan=plan,
        value=value,
        values=(Float32(3), Float32(5)),
        value_type=Float32,
        op="sum",
        initial_value=None,
        aggregate_output=aggregate,
        valid_items=None,
        deferred_temp_storage=None,
    )

    assert len(tensors) == 2
    assert ffi_calls[0][-2:] == (
        tensors[0].iterator.llvm_ptr,
        tensors[1].iterator.llvm_ptr,
    )

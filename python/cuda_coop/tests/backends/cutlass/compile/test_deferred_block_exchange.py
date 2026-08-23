# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

cutlass_common = pytest.importorskip("cutlass.base_dsl.common")
cutlass_memory = pytest.importorskip("cutlass.memory")
cutlass_typing = pytest.importorskip("cutlass.base_dsl.typing")
coop = pytest.importorskip("cuda.coop.cutlass")
coop_core = pytest.importorskip("cuda.coop._core")
coop_core_block = pytest.importorskip("cuda.coop._core.block")
exchange_provider = pytest.importorskip("cuda.coop.cutlass._dsl._cub_exchange_provider")
block_provider = pytest.importorskip("cuda.coop.cutlass._dsl.block._provider")
provider_support = pytest.importorskip("cuda.coop.cutlass._dsl._provider")
provider_bundle = pytest.importorskip("cuda.coop.cutlass._dsl._provider_bundle")
single_phase = pytest.importorskip("cuda.coop.cutlass._dsl._single_phase")

DSLRuntimeError = cutlass_common.DSLRuntimeError
Float32 = cutlass_typing.Float32
LaunchFacts = coop_core.LaunchFacts
BlockExchangeMode = coop_core_block.BlockExchangeMode


class _FakeKernel:
    def __init__(self, name: str):
        self.name = name


class _Placeholder:
    def __init__(self):
        self.replacements = []

    def replace_all_uses_with(self, replacement):
        self.replacements.append(replacement)


def _event(*, kernel, storage, requirement_key):
    return provider_support.DeferredTempStorageEvent(
        kernel_op=kernel,
        kernel_name=kernel.name,
        temp_storage=storage,
        primitive_name="exchange",
        requirement_key=requirement_key,
        sharing=storage.sharing,
        auto_sync=storage.auto_sync,
        capacity_size_in_bytes=storage.capacity_size_in_bytes,
        capacity_alignment=storage.alignment,
        smem_addr_placeholder=_Placeholder(),
        size_placeholder=_Placeholder(),
        location=f"test:{kernel.name}:exchange",
    )


def _exchange_request(*, external_scratch: bool):
    return exchange_provider._make_request(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        value_type=Float32,
        items_per_thread=4,
        mode=BlockExchangeMode.BLOCKED_TO_STRIPED,
        rank_type=None,
        valid_flag_type=None,
        warp_time_slicing=False,
        source="scoped_block",
        external_scratch=external_scratch,
    )


def test_deferred_temp_storage_is_automatic_and_manual():
    explicit = coop._block.TempStorage(size_in_bytes=256)
    assert not explicit.is_deferred
    assert explicit.auto_sync

    explicit_manual = coop._block.TempStorage(
        size_in_bytes=256,
        auto_sync=False,
    )
    assert not explicit_manual.is_deferred
    assert not explicit_manual.auto_sync
    inferred = coop._block.TempStorage()
    assert inferred.is_deferred
    assert inferred.capacity_size_in_bytes is None
    assert inferred.alignment is None
    assert inferred.auto_sync

    inferred_manual = coop._block.TempStorage(auto_sync=False)
    assert inferred_manual.is_deferred
    assert not inferred_manual.auto_sync

    inferred_automatic = coop._block.TempStorage(auto_sync=True)
    assert inferred_automatic.auto_sync


def test_deferred_temp_storage_requires_trace_finalize_capability():
    dsl_without_finalize = type("Dsl", (), {"compile_options": object()})()

    with pytest.raises(
        DSLRuntimeError,
        match=r"(?s)trace-finalize hook.*runtime\s+separately",
    ):
        provider_support.ensure_trace_hook_registered(
            get_cute_dsl=lambda: dsl_without_finalize,
        )


def test_deferred_temp_storage_requires_allocation_capabilities(monkeypatch):
    monkeypatch.setattr(cutlass_memory, "SmemAllocator", object)

    with pytest.raises(DSLRuntimeError, match=r"runtime\s+separately"):
        provider_support.materialize_deferred_temp_storage_plans(
            (object(),),
            object(),
        )


def test_managed_link_cleanup_allows_fresh_compile_options(monkeypatch, tmp_path):
    managed_path = str(tmp_path / "managed.ltoir")
    monkeypatch.setattr(
        provider_bundle,
        "managed_bundle_paths",
        lambda: frozenset({managed_path}),
    )
    dsl = type(
        "Dsl",
        (),
        {"compile_options": type("Options", (), {"options": {}})()},
    )()

    block_provider._remove_managed_bundle_link_options(dsl)

    assert dsl.compile_options.options == {}


def test_deferred_temp_storage_rejects_non_exchange_provider(monkeypatch):
    storage = coop._block.TempStorage()
    context = single_phase.SinglePhaseContext(
        thread_data=None,
        temp_storage=storage,
    )

    with single_phase.activate_single_phase_context(context):
        with pytest.raises(
            NotImplementedError,
            match=(
                r"deferred TempStorage.*block Load, Store, Exchange, Scan, "
                r"AdjacentDifference, Discontinuity, RadixSort, and MergeSort"
            ),
        ):
            provider_support.temp_storage_ffi_args(
                "sum",
                active_session_getter=provider_support.BundleSession,
            )


def test_deferred_planner_strengthens_and_isolates_identity(monkeypatch):
    first_kernel = _FakeKernel("same_name")
    second_kernel = _FakeKernel("same_name")
    shared_storage = coop._block.TempStorage()
    other_storage = coop._block.TempStorage()

    events = [
        _event(kernel=first_kernel, storage=shared_storage, requirement_key="f32"),
        _event(kernel=first_kernel, storage=shared_storage, requirement_key="f64"),
        _event(kernel=first_kernel, storage=other_storage, requirement_key="f32"),
        _event(kernel=second_kernel, storage=shared_storage, requirement_key="f32"),
    ]
    layouts = {
        "f32": provider_support.ScratchLayout(1024, 16),
        "f64": provider_support.ScratchLayout(2048, 16),
    }

    plans = provider_support.plan_deferred_temp_storage_events(events, layouts)

    assert len(plans) == 3
    strengthened = next(
        plan
        for plan in plans
        if plan.kernel_op is first_kernel and plan.temp_storage is shared_storage
    )
    assert strengthened.size_in_bytes == 2048
    assert strengthened.alignment == 16
    assert [binding.size_in_bytes for binding in strengthened.bindings] == [
        2048,
        2048,
    ]
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


def test_deferred_planner_validation_does_not_touch_placeholders(monkeypatch):
    event = _event(
        kernel=_FakeKernel("kernel"),
        storage=coop._block.TempStorage(),
        requirement_key="missing",
    )

    with pytest.raises(DSLRuntimeError, match=r"No exact C\+\+ scratch layout"):
        provider_support.plan_deferred_temp_storage_events([event], {})

    assert event.smem_addr_placeholder.replacements == []
    assert event.size_placeholder.replacements == []


def test_materializer_uses_one_allocator_per_kernel(monkeypatch):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith, llvm

    allocator_instances = []

    class _Pointer:
        def to_llvm_ptr(self):
            return self

    class _Allocator:
        def __init__(self):
            self.allocations = []
            allocator_instances.append(self)

        def allocate(self, size_in_bytes, alignment):
            self.allocations.append((size_in_bytes, alignment))
            return _Pointer()

    monkeypatch.setattr(cutlass_memory, "SmemAllocator", _Allocator)
    monkeypatch.setattr(llvm, "ptrtoint", lambda *_args: object())
    monkeypatch.setattr(arith, "constant", lambda *_args: object())
    monkeypatch.setattr(arith, "addi", lambda *_args: object())

    def make_plan(kernel_op, kernel_name, size_in_bytes):
        address = _Placeholder()
        size = _Placeholder()
        storage = object()
        event = provider_support.DeferredTempStorageEvent(
            kernel_op=kernel_op,
            kernel_name=kernel_name,
            temp_storage=storage,
            primitive_name="exchange",
            requirement_key=(kernel_name, size_in_bytes),
            sharing="shared",
            auto_sync=False,
            capacity_size_in_bytes=None,
            capacity_alignment=None,
            smem_addr_placeholder=address,
            size_placeholder=size,
            location=f"test:{kernel_name}",
        )
        binding = provider_support.DeferredTempStorageBinding(
            event=event,
            byte_offset_in_bytes=0,
            size_in_bytes=size_in_bytes,
            alignment=16,
        )
        return provider_support.DeferredTempStoragePlan(
            kernel_op=kernel_op,
            kernel_name=kernel_name,
            temp_storage=storage,
            size_in_bytes=size_in_bytes,
            alignment=16,
            bindings=(binding,),
        )

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(
            "module { func.func @first() { return } func.func @second() { return } }"
        )
        first_kernel = module.body.operations[0]
        same_first_kernel = module.body.operations[0]
        second_kernel = module.body.operations[1]
        assert same_first_kernel is not first_kernel
        assert same_first_kernel == first_kernel
        plans = (
            make_plan(first_kernel, "first", 1024),
            make_plan(same_first_kernel, "first", 2048),
            make_plan(second_kernel, "second", 512),
        )

        provider_support.materialize_deferred_temp_storage_plans(plans, module)

        assert [allocator.allocations for allocator in allocator_instances] == [
            [(1024, 16), (2048, 16)],
            [(512, 16)],
        ]
        assert all(
            len(binding.event.smem_addr_placeholder.replacements) == 1
            and len(binding.event.size_placeholder.replacements) == 1
            for plan in plans
            for binding in plan.bindings
        )


def test_static_and_deferred_exchange_requests_have_distinct_abis():
    static_request = _exchange_request(external_scratch=False)
    deferred_request = _exchange_request(external_scratch=True)

    assert static_request != deferred_request
    assert hash(static_request) != hash(deferred_request)
    assert len({static_request, deferred_request}) == 2
    assert (
        deferred_request.symbol_name == f"{static_request.symbol_name}_external_scratch"
    )

    static_source = "\n".join(exchange_provider._render_cub_exchange(static_request))
    deferred_source = "\n".join(
        exchange_provider._render_cub_exchange(deferred_request)
    )

    assert (
        "__shared__ typename implementation_type::TempStorage storage;" in static_source
    )
    assert "cuda_coop_cutlass_block_sync();" in static_source
    assert "unsigned int temp_storage_smem_addr" in deferred_source
    assert "int temp_storage_bytes" in deferred_source
    assert "int temp_storage_auto_sync" in deferred_source
    assert (
        "typename implementation_type::TempStorage*>(temp_storage_ptr)"
        in deferred_source
    )
    assert "__shared__ typename implementation_type::TempStorage" not in deferred_source
    assert "if (temp_storage_auto_sync != 0)" in deferred_source
    assert "cuda_coop_cutlass_block_sync();" in deferred_source

    probe = exchange_provider._cub_exchange_scratch_layout_probe(deferred_request)
    assert probe is not None
    assert probe.requirement_key == deferred_request.scratch_requirement_key
    assert probe.size_expression == f"sizeof({deferred_request.scratch_cpp_type})"
    assert probe.alignment_expression == f"alignof({deferred_request.scratch_cpp_type})"

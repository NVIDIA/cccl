# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import pytest


def _provider_dependencies():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")


def _launch_facts(block_dim=64):
    from cuda.coop._core import LaunchFactOrigin, LaunchFacts

    return LaunchFacts(
        exact_block_dim=block_dim,
        provenance=LaunchFactOrigin(
            "exact_block_dim",
            "test_kernel",
            verified=True,
        ),
    )


def test_public_data_movement_exports_and_signatures():
    _provider_dependencies()

    import cuda.coop.cutlass as coop

    expected_modules = {
        "load": "cuda.coop.cutlass._group_load_store",
        "store": "cuda.coop.cutlass._group_load_store",
    }
    for name, module in expected_modules.items():
        assert name in coop.__all__
        assert getattr(coop, name).__module__ == module
        assert all(
            not parameter.startswith("_")
            for parameter in inspect.signature(getattr(coop, name)).parameters
        )


def test_frontends_delegate_resolved_groups_and_plans(monkeypatch):
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import _load_store as load_store_provider

    monkeypatch.setattr(_launch, "current_kernel_launch_facts", _launch_facts)
    calls = []
    monkeypatch.setattr(
        load_store_provider,
        "provider_load",
        lambda **payload: calls.append(("load", payload)) or payload["output"],
    )
    monkeypatch.setattr(
        load_store_provider,
        "provider_store",
        lambda **payload: calls.append(("store", payload)),
    )

    block = coop.this_block()
    output = coop.ThreadData(2, dtype=Int32)
    items = coop.ThreadData.from_values(Int32(1), Int32(2), dtype=Int32)
    assert coop.load(block, object(), output, valid_items=17, offset=4) is output
    coop.store(block, object(), items, algorithm="striped")
    assert [name for name, _ in calls] == ["load", "store"]
    for _, payload in calls:
        group = payload["group"]
        assert group.hierarchy.block_dim == (64, 1, 1)
        assert group is not block


def test_load_store_plans_bindings_rendering_and_layout_proofs():
    _provider_dependencies()
    from cutlass.base_dsl.typing import Float32, Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import ArgumentBinding, GroupLoadStoreKind, LaunchFacts
    from cuda.coop.cutlass import _group_load_store as frontend
    from cuda.coop.cutlass._lowering import _load_store as provider
    from cuda.coop.cutlass._lowering._load_store_layout import (
        contiguous_layout_reason,
        static_layout_elements,
    )

    block_plan = provider._make_group_load_store_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind.LOAD,
        dtype=Int32,
        items_per_thread=2,
        algorithm="striped",
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(0),
        offset=ArgumentBinding.static(4),
    ).require_supported()
    block_request = provider._CubLoadStoreRequest(block_plan, Int32)
    block_source = "\n".join(provider._render_cub_load_store(block_request))
    assert "::cub::BlockLoad<int, 64, 2, ::cub::BLOCK_LOAD_STRIPED" in block_source
    assert "const int* base, int valid_items, int* result_items" in block_source
    assert "tile_ptr += 4;" in block_source
    assert ".Load(tile_ptr, items, valid_items, static_cast<int>(0));" in block_source
    assert "cuda_coop_cutlass_block_sync();" in block_source

    partial_block_plan = provider._make_group_load_store_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind.LOAD,
        dtype=Int32,
        items_per_thread=2,
        algorithm="striped",
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.static(4),
    ).require_supported()
    partial_block_source = "\n".join(
        provider._render_cub_load_store(
            provider._CubLoadStoreRequest(partial_block_plan, Int32)
        )
    )
    assert "int items[2] = {};" in partial_block_source
    assert ".Load(tile_ptr, items, valid_items);" in partial_block_source

    partial_warp_plan = provider._make_group_load_store_plan(
        group=coop.this_warp().group_by(8),
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind.LOAD,
        dtype=Int32,
        items_per_thread=2,
        algorithm="striped",
        valid_items=ArgumentBinding.static(5),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.omitted(),
    ).require_supported()
    partial_warp_source = "\n".join(
        provider._render_cub_load_store(
            provider._CubLoadStoreRequest(partial_warp_plan, Int32)
        )
    )
    assert "int items[2] = {};" in partial_warp_source
    assert ".Load(tile_ptr, items, 5);" in partial_warp_source

    warp_plan = provider._make_group_load_store_plan(
        group=coop.this_warp(),
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind.STORE,
        dtype=Float32,
        items_per_thread=3,
        algorithm="transpose",
        valid_items=ArgumentBinding.omitted(),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.runtime(),
    ).require_supported()
    warp_request = provider._CubLoadStoreRequest(warp_plan, Float32)
    warp_source = "\n".join(provider._render_cub_load_store(warp_request))
    assert "::cub::WarpStore<float, 3, ::cub::WARP_STORE_TRANSPOSE, 32>" in warp_source
    assert "storage[2]" in warp_source
    assert "storage_instance) * 96ll" in warp_source
    assert "cuda_coop_cutlass_warp_sync(32u);" in warp_source
    assert "if (offset < 0)" in warp_source
    assert warp_source.index("if (offset < 0)") < warp_source.index(
        "tile_ptr += offset;"
    )

    assert frontend._classify_integer_binding(3, name="offset") == (
        ArgumentBinding.static(3)
    )
    assert frontend._classify_oob_default(1.25) == ArgumentBinding.static(1.25)
    assert frontend._classify_oob_default(Float32(1.25)) == (ArgumentBinding.runtime())

    import numpy as np

    numpy_integer = frontend._classify_oob_default(np.int64(7))
    numpy_float = frontend._classify_oob_default(np.float32(1.25))
    assert numpy_integer == ArgumentBinding.static(7)
    assert numpy_float == ArgumentBinding.static(1.25)
    assert type(numpy_integer.value) is int
    assert type(numpy_float.value) is float
    with pytest.raises(TypeError, match="must be an integer"):
        frontend._classify_integer_binding(True, name="valid_items")
    with pytest.raises(ValueError, match="non-negative"):
        frontend._classify_integer_binding(-1, name="offset")

    class Layout:
        def __init__(self, shape, stride):
            self.shape = shape
            self.stride = stride

    compact = Layout(((8, 4), 2), ((1, 8), 32))
    assert static_layout_elements(compact) == 64
    assert contiguous_layout_reason(compact) is None
    scalar = Layout((), ())
    assert static_layout_elements(scalar) == 1
    assert contiguous_layout_reason(scalar) is None
    singleton_mode = Layout((1, 8), (0, 1))
    assert static_layout_elements(singleton_mode) == 8
    assert contiguous_layout_reason(singleton_mode) is None
    assert "not a compact" in contiguous_layout_reason(Layout((8, 4, 2), (12, 3, 1)))
    assert "incongruent" in contiguous_layout_reason(Layout(((8, 4), 2), (1, 8)))


def test_deferred_load_store_requests_do_not_collide():
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import (
        ArgumentBinding,
        GroupLoadStoreKind,
        LaunchFacts,
        StorageOwnership,
    )
    from cuda.coop.cutlass._lowering import _load_store as provider

    plan = provider._make_group_load_store_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind.LOAD,
        dtype=Int32,
        items_per_thread=4,
        algorithm="transpose",
        valid_items=ArgumentBinding.omitted(),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.omitted(),
    ).require_supported()
    assert plan.temp_storage is not None
    caller_owned = dataclasses.replace(
        plan,
        temp_storage=dataclasses.replace(
            plan.temp_storage,
            ownership=StorageOwnership.CALLER,
            address_space="shared",
            cpp_type="typename implementation_type::TempStorage",
            instances=1,
            instance_index="cta",
            exact_layout_required=True,
        ),
    )
    owned = provider._CubLoadStoreRequest(plan, Int32)
    deferred = provider._CubLoadStoreRequest(
        caller_owned,
        Int32,
        external_scratch=True,
    )
    assert owned != deferred
    assert owned.symbol_name != deferred.symbol_name
    assert deferred.symbol_name == f"{owned.symbol_name}_external_scratch"
    assert provider._cub_load_store_scratch_layout_probe(owned) is None
    probe = provider._cub_load_store_scratch_layout_probe(deferred)
    assert probe is not None
    assert probe.requirement_key == deferred.scratch_requirement_key


def test_core_output_initializers_require_a_readable_source():
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import (
        ArgumentBinding,
        Array,
        Dependency,
        GroupLoadStoreKind,
        LaunchFacts,
        TempStorageParameter,
    )
    from cuda.coop.cutlass._lowering import _load_store as provider
    from cuda.coop.cutlass._lowering._core import CutlassCoreAdapter

    plan = provider._make_group_load_store_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind.LOAD,
        dtype=Int32,
        items_per_thread=2,
        algorithm="striped",
        valid_items=ArgumentBinding.omitted(),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.omitted(),
    ).require_supported()

    def specialization_with_source(*, is_output: bool, is_inout: bool = False):
        parameters = (
            (
                TempStorageParameter(),
                Array(
                    Dependency("T"),
                    Dependency("ITEMS_PER_THREAD"),
                    name="src",
                    is_output=is_output,
                    is_inout=is_inout,
                ),
                Array(
                    Dependency("T"),
                    Dependency("ITEMS_PER_THREAD"),
                    name="dst",
                    is_output=True,
                ),
            ),
        )
        algorithm = dataclasses.replace(
            plan.implementation.algorithm,
            parameters=parameters,
        )
        return dataclasses.replace(plan.implementation, algorithm=algorithm)

    output_only = specialization_with_source(is_output=True)
    with pytest.raises(ValueError, match="source 'src' is output-only"):
        CutlassCoreAdapter().materialize(
            output_only,
            plan=dataclasses.replace(plan, implementation=output_only),
            kind="output_initializer_test",
            output_initializers=(("dst", "src"),),
        )

    inout = specialization_with_source(is_output=True, is_inout=True)
    artifact = CutlassCoreAdapter().materialize(
        inout,
        plan=dataclasses.replace(plan, implementation=inout),
        kind="output_initializer_test",
        output_initializers=(("dst", "src"),),
    )
    assert artifact.output_initializers == (("dst", "src"),)


def test_core_artifact_identity_includes_renderer_kind_and_method_index():
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import ArgumentBinding, GroupLoadStoreKind, LaunchFacts
    from cuda.coop.cutlass._compiler._rendering import canonical_bundle_requests
    from cuda.coop.cutlass._lowering import _load_store as provider
    from cuda.coop.cutlass._lowering._core import CutlassCoreAdapter

    plan = provider._make_group_load_store_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind.LOAD,
        dtype=Int32,
        items_per_thread=2,
        algorithm="striped",
        valid_items=ArgumentBinding.omitted(),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.omitted(),
    ).require_supported()
    adapter = CutlassCoreAdapter()

    first = adapter.materialize(
        plan.implementation,
        plan=plan,
        kind="first_renderer",
    )
    second = adapter.materialize(
        plan.implementation,
        plan=plan,
        kind="second_renderer",
    )
    assert first.semantic_key != second.semantic_key
    assert first.symbol_name != second.symbol_name

    first = dataclasses.replace(first, symbol_name="shared_symbol")
    second = dataclasses.replace(second, symbol_name="shared_symbol")
    with pytest.raises(ValueError, match="conflicting bundle requests"):
        canonical_bundle_requests((first, second))

    with pytest.raises(ValueError, match="method_index is out of range"):
        adapter.materialize(
            plan.implementation,
            plan=plan,
            kind="negative_method",
            method_index=-1,
        )


def test_load_store_rejects_register_and_local_address_space_pointers(monkeypatch):
    _provider_dependencies()
    from cutlass import AddressSpace

    from cuda.coop.cutlass._lowering import _load_store as provider

    class FakePointer:
        def __init__(self, address_space):
            self.type = SimpleNamespace(address_space=address_space)

    class Operand:
        shape = (1,)
        stride = (1,)

        def __init__(self, address_space, *, memspace=None):
            self.iterator = SimpleNamespace(llvm_ptr=FakePointer(address_space))
            if memspace is not None:
                self.memspace = memspace

    class FakePointerType:
        def __new__(cls, pointer_type):
            return pointer_type

        @staticmethod
        def get(address_space):
            return SimpleNamespace(address_space=address_space)

    monkeypatch.setattr(provider.llvm, "PointerType", FakePointerType)
    monkeypatch.setattr(
        provider.llvm,
        "addrspacecast",
        lambda pointer_type, pointer: (pointer_type, pointer),
    )

    proof, reason = provider._contiguous_memory_proof(
        Operand(0, memspace=AddressSpace.rmem),
        primitive_name="load",
    )
    assert proof is None
    assert "register/local memory" in reason

    proof, _ = provider._contiguous_memory_proof(
        Operand(5),
        primitive_name="load",
    )
    assert proof is None

    for address_space, memspace in (
        (1, AddressSpace.gmem),
        (3, AddressSpace.smem),
    ):
        proof, reason = provider._contiguous_memory_proof(
            Operand(address_space, memspace=memspace),
            primitive_name="load",
        )
        assert proof is not None, reason


def test_fixed_load_store_storage_is_forwarded_and_scoped(monkeypatch):
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._compiler import _storage as provider_storage
    from cuda.coop.cutlass._lowering import _load_store as provider

    storage = coop.TempStorage(4096, alignment=16, auto_sync=False)
    assert (
        provider._temp_storage_for_load_store(
            group=coop.this_block(),
            explicit_temp_storage=storage,
            primitive_name="load",
        )
        is storage
    )
    for warp_storage in (storage, coop.TempStorage()):
        with pytest.raises(ValueError, match="only for block groups"):
            provider._temp_storage_for_load_store(
                group=coop.this_warp(),
                explicit_temp_storage=warp_storage,
                primitive_name="load",
            )
    with pytest.raises(ValueError, match="sharing='exclusive'"):
        provider._temp_storage_for_load_store(
            group=coop.this_block(),
            explicit_temp_storage=coop.TempStorage(4096, sharing="exclusive"),
            primitive_name="store",
        )

    monkeypatch.setattr(
        provider_storage,
        "materialize_temp_storage_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            smem_addr_u32="shared-address",
            size_in_bytes=4096,
            auto_sync=False,
        ),
    )
    assert provider._external_scratch_args(
        storage,
        primitive_name="load",
        requirement_key=("load", "int32"),
    ) == ("shared-address", Int32(4096), Int32(0))


class _FakeTensor:
    class _Iterator:
        llvm_ptr = object()

    iterator = _Iterator()

    def __getitem__(self, index):
        return index


def test_deferred_load_failure_rolls_back_provider_session(monkeypatch):
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    import cuda.coop.cutlass._compiler._state as provider_state
    import cuda.coop.cutlass._compiler._storage as provider_storage
    from cuda.coop._core import ArgumentBinding, GroupLoadStoreAlgorithm, LaunchFacts
    from cuda.coop.cutlass._lowering import _load_store as provider

    storage = coop.TempStorage()
    snapshot = object()
    restored = []
    monkeypatch.setattr(provider, "_resolve_memory_type", lambda *args, **kwargs: Int32)
    monkeypatch.setattr(provider, "_memory_pointer", lambda *args, **kwargs: object())
    monkeypatch.setattr(provider._cute, "make_rmem_tensor", lambda *args: _FakeTensor())
    monkeypatch.setattr(
        provider_state, "snapshot_active_session_state", lambda: snapshot
    )
    monkeypatch.setattr(provider_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(provider_state, "register_request", lambda _request: None)
    monkeypatch.setattr(
        provider_storage,
        "register_deferred_temp_storage_event",
        lambda *args, **kwargs: (object(), object(), object()),
    )
    monkeypatch.setattr(provider.llvm.PointerType, "get", lambda *_args: object())

    def failing_ffi(**_kwargs):
        def invoke(*_args):
            raise RuntimeError("forced FFI failure")

        return invoke

    monkeypatch.setattr(provider, "ffi", failing_ffi)
    with pytest.raises(RuntimeError, match="forced FFI failure"):
        provider.provider_load(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=64),
            source=object(),
            output=coop.ThreadData(4, dtype=Int32),
            algorithm=GroupLoadStoreAlgorithm.TRANSPOSE,
            valid_items=None,
            valid_items_binding=ArgumentBinding.omitted(),
            oob_default=None,
            oob_default_binding=ArgumentBinding.omitted(),
            offset=None,
            offset_binding=ArgumentBinding.omitted(),
            temp_storage=storage,
        )

    assert restored == [snapshot]

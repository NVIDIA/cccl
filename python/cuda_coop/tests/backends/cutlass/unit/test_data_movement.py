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
        "exchange": "cuda.coop.cutlass._group_exchange",
        "load": "cuda.coop.cutlass._group_load_store",
        "shuffle": "cuda.coop.cutlass._group_shuffle",
        "store": "cuda.coop.cutlass._group_load_store",
    }
    for name, module in expected_modules.items():
        assert name in coop.__all__
        assert getattr(coop, name).__module__ == module
        assert all(
            not parameter.startswith("_")
            for parameter in inspect.signature(getattr(coop, name)).parameters
        )
    assert "output" not in inspect.signature(coop.exchange).parameters
    assert "temp_storage" not in inspect.signature(coop.exchange).parameters


def test_frontends_delegate_resolved_groups_and_plans(monkeypatch):
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringPlan
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import _exchange as exchange_provider
    from cuda.coop.cutlass._lowering import _load_store as load_store_provider
    from cuda.coop.cutlass._lowering import _shuffle as shuffle_provider

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
    monkeypatch.setattr(
        exchange_provider,
        "provider_exchange",
        lambda **payload: calls.append(("exchange", payload)) or payload["value"],
    )
    monkeypatch.setattr(
        shuffle_provider,
        "provider_shuffle",
        lambda **payload: calls.append(("shuffle", payload)) or payload["value"],
    )

    block = coop.this_block()
    output = coop.ThreadData(2, dtype=Int32)
    items = coop.ThreadData.from_values(Int32(1), Int32(2), dtype=Int32)
    assert coop.load(block, object(), output, valid_items=17, offset=4) is output
    coop.store(block, object(), items, algorithm="striped")
    assert coop.exchange(block, items, mode="blocked_to_striped") is items
    assert coop.shuffle(block, items, mode="down") is items

    assert [name for name, _ in calls] == ["load", "store", "exchange", "shuffle"]
    for name, payload in calls:
        if name == "exchange":
            assert isinstance(payload["plan"], GroupLoweringPlan)
            group = payload["plan"].resolved_group
            assert payload.keys() == {"plan", "value", "ranks", "valid_flags"}
        else:
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
    assert "cuda_coop_cutlass_warp_sync();" in warp_source
    assert "if (offset < 0)" in warp_source
    assert warp_source.index("if (offset < 0)") < warp_source.index(
        "tile_ptr += offset;"
    )

    assert frontend._classify_integer_binding(3, name="offset") == (
        ArgumentBinding.static(3)
    )
    assert frontend._classify_oob_default(1.25) == ArgumentBinding.static(1.25)
    assert frontend._classify_oob_default(Float32(1.25)) == (ArgumentBinding.runtime())
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
    assert "not a compact" in contiguous_layout_reason(Layout((8, 4, 2), (12, 3, 1)))
    assert "incongruent" in contiguous_layout_reason(Layout(((8, 4), 2), (1, 8)))


def test_exchange_requests_render_block_logical_warp_ranks_and_flags():
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32, Uint32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._lowering import _exchange as provider

    block_plan = provider._make_group_exchange_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        items_per_thread=2,
        mode="scatter_to_striped_flagged",
        rank_dtype=Int32,
        valid_flag_dtype=Uint32,
    ).require_supported()
    block_request = provider._CubExchangeRequest(
        block_plan,
        Int32,
        rank_type=Int32,
        valid_flag_type=Uint32,
    )
    block_source = "\n".join(provider._render_cub_exchange(block_request))
    assert "::cub::BlockExchange<int, 64, 2, 0, 1, 1>" in block_source
    assert ".ScatterToStripedFlagged(" in block_source
    assert "input_items, output_items, ranks, valid_flags" in block_source
    assert "external_scratch" not in block_source

    logical_plan = provider._make_group_exchange_plan(
        group=coop.this_warp().group_by(8),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        items_per_thread=2,
        mode="scatter_to_striped",
        rank_dtype=Int32,
    ).require_supported()
    logical_request = provider._CubExchangeRequest(
        logical_plan,
        Int32,
        rank_type=Int32,
    )
    logical_source = "\n".join(provider._render_cub_exchange(logical_request))
    assert "::cub::WarpExchange<int, 2, 8" in logical_source
    assert "TempStorage storage[8]" in logical_source
    assert "cuda_coop_cutlass_linear_tid() / 8u" in logical_source
    assert "cuda_coop_cutlass_warp_sync();" in logical_source

    with pytest.raises(TypeError, match="GroupLoweringPlan"):
        provider._CubExchangeRequest(None, Int32)


def test_exchange_symbols_distinguish_lowering_plans():
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._lowering import _exchange as provider

    def request(group, *, warp_time_slicing=False):
        plan = provider._make_group_exchange_plan(
            group=group,
            launch=LaunchFacts(exact_block_dim=64),
            dtype=Int32,
            items_per_thread=2,
            mode="blocked_to_striped",
            warp_time_slicing=warp_time_slicing,
        ).require_supported()
        return provider._CubExchangeRequest(plan, Int32)

    block = request(coop.this_block())
    time_sliced_block = request(coop.this_block(), warp_time_slicing=True)
    logical_warp_8 = request(coop.this_warp().group_by(8))
    logical_warp_16 = request(coop.this_warp().group_by(16))

    assert (
        len(
            {
                block.symbol_name,
                time_sliced_block.symbol_name,
                logical_warp_8.symbol_name,
                logical_warp_16.symbol_name,
            }
        )
        == 4
    )


def test_shuffle_routes_and_renderers():
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockShuffleMode
    from cuda.coop.cutlass import _group_shuffle as frontend
    from cuda.coop.cutlass._lowering import _shuffle as provider
    from cuda.coop.cutlass._lowering._core import render_cutlass_core_artifact

    request = provider._make_request(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        value_type=Int32,
        items_per_thread=4,
        mode=BlockShuffleMode.UP,
        block_prefix=False,
        block_suffix=True,
    )
    source = "\n".join(render_cutlass_core_artifact(request))
    assert "::cub::BlockShuffle<int, 8, 4, 2>" in source
    assert ".Up(input_items, output_items, *block_suffix)" in source

    items = coop.ThreadData.from_values(Int32(1), Int32(2), dtype=Int32)
    with pytest.raises(NotImplementedError, match="only distance=1"):
        frontend._normalize_shuffle_route(
            items,
            mode=BlockShuffleMode.UP,
            distance=2,
            block_prefix=None,
            block_suffix=None,
        )
    with pytest.raises(NotImplementedError, match="only public-CUB Up/Down"):
        frontend._normalize_shuffle_route(
            items,
            mode=BlockShuffleMode.ROTATE,
            distance=1,
            block_prefix=None,
            block_suffix=None,
        )


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
    from cuda.coop._core import ArgumentBinding, GroupLoadStoreAlgorithm, LaunchFacts
    from cuda.coop.cutlass._compiler import _state as provider_state
    from cuda.coop.cutlass._compiler import _storage as provider_storage
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

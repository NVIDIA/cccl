# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT


def _radix_request(*, external_scratch: bool):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import GroupOperandKind, LaunchFacts
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider

    return provider._make_sort_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        descending=False,
        source="cutlass_root",
        external_scratch=external_scratch,
    )


def _radix_pair_request(*, external_scratch: bool):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Float64, Int32

    from cuda.coop._core import GroupOperandKind, LaunchFacts
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider

    return provider._make_sort_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=Float64,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        descending=True,
        source="cutlass_root",
        external_scratch=external_scratch,
    )


def _merge_request(*, external_scratch: bool):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Float64, Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider

    return provider._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=Float64,
        items_per_thread=2,
        descending=True,
        valid_items=None,
        oob_default=None,
        source="cutlass_root",
        external_scratch=external_scratch,
    )


def _merge_key_request(*, external_scratch: bool):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider

    return provider._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=2,
        descending=False,
        valid_items=None,
        oob_default=None,
        source="cutlass_root",
        external_scratch=external_scratch,
    )


def _adjacent_difference_request(
    *,
    external_scratch: bool,
    source: str = "cutlass_root",
):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockAdjacentDifferenceDirection
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_adjacent_difference_provider as provider

    return provider._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        value_type=Int32,
        items_per_thread=2,
        direction=BlockAdjacentDifferenceDirection.LEFT,
        valid_items=None,
        tile_predecessor_item=None,
        tile_successor_item=None,
        source=source,
        external_scratch=external_scratch,
    )


def _discontinuity_request(
    *,
    external_scratch: bool,
    source: str = "cutlass_root",
):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockDiscontinuityMode
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_discontinuity_provider as provider

    return provider._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        value_type=Int32,
        items_per_thread=2,
        mode=BlockDiscontinuityMode.HEADS,
        tile_predecessor_item=None,
        tile_successor_item=None,
        source=source,
        external_scratch=external_scratch,
    )


@pytest.mark.parametrize(
    "request_factory",
    [
        _radix_request,
        _radix_pair_request,
        _merge_key_request,
        _merge_request,
        _adjacent_difference_request,
        _discontinuity_request,
    ],
)
def test_core_external_scratch_is_opt_in_and_preserves_default_abi(request_factory):
    from cuda.coop._core import StorageOwnership
    from cuda.coop.cutlass._dsl import _provider as provider_support
    from cuda.coop.cutlass._dsl._core_adapter import render_cutlass_core_artifact

    owned = request_factory(external_scratch=False)
    external = request_factory(external_scratch=True)

    assert owned.plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert external.plan.temp_storage.ownership is StorageOwnership.CALLER
    assert owned.semantic_key != external.semantic_key
    assert not owned.symbol_name.endswith("_external_scratch")
    assert external.symbol_name.endswith("_external_scratch")
    assert external.abi_parameters == owned.abi_parameters

    owned_source = "\n".join(render_cutlass_core_artifact(owned))
    external_source = "\n".join(render_cutlass_core_artifact(external))
    assert "__shared__ typename implementation_type::TempStorage storage;" in (
        owned_source
    )
    assert "cuda_coop_cutlass_block_sync();" in owned_source
    assert "unsigned int temp_storage_smem_addr" not in owned_source
    assert "__shared__ typename implementation_type::TempStorage storage;" not in (
        external_source
    )
    assert "unsigned int temp_storage_smem_addr" in external_source
    assert "int temp_storage_bytes" in external_source
    assert "int temp_storage_auto_sync" in external_source
    assert "implementation_type(*storage_ptr)" in external_source
    assert "if (temp_storage_auto_sync != 0)" in external_source
    assert "cuda_coop_cutlass_block_sync();" in external_source

    renderer = provider_support.bundle_renderer_for(external)
    assert renderer is not None
    assert renderer.scratch_layout_probe is not None
    assert renderer.scratch_layout_probe(owned) is None
    probe = renderer.scratch_layout_probe(external)
    assert probe is not None
    assert probe.requirement_key == external.scratch_requirement_key
    assert probe.size_expression == f"sizeof({external.scratch_cpp_type})"
    assert probe.alignment_expression == f"alignof({external.scratch_cpp_type})"


def test_core_external_scratch_binding_appends_fixed_i32_operands():
    request = _radix_request(external_scratch=True)

    arguments = request.bind_ffi_arguments(
        {"keys": (7, 3), "begin_bit": 0, "end_bit": 8},
        {"keys": "result-pointer"},
        scratch_values=("smem-address", "smem-size", "auto-sync"),
    )
    assert arguments[-3:] == ("smem-address", "smem-size", "auto-sync")
    with pytest.raises(ValueError, match="expected 3 scratch ABI values"):
        request.bind_ffi_arguments(
            {"keys": (7, 3), "begin_bit": 0, "end_bit": 8},
            {"keys": "result-pointer"},
        )


def test_core_external_scratch_rejects_warp_merge_sort():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import this_warp
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider

    with pytest.raises(ValueError, match="block-scoped only"):
        provider._make_request(
            group=this_warp(),
            launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
            key_type=Int32,
            value_type=None,
            items_per_thread=2,
            descending=False,
            valid_items=None,
            oob_default=None,
            source="cutlass_root",
            external_scratch=True,
        )


def test_scoped_radix_and_merge_sort_adapters_advertise_deferred_storage():
    pytest.importorskip("cutlass")
    from cuda.coop.cutlass._dsl.block import _sort

    assert _sort._radix_sort_keys_provider._supports_deferred_temp_storage
    assert _sort._radix_sort_pairs_provider._supports_deferred_temp_storage
    assert _sort._merge_sort_keys_provider._supports_deferred_temp_storage
    assert _sort._merge_sort_pairs_provider._supports_deferred_temp_storage
    assert not hasattr(
        _sort._radix_rank_provider,
        "_supports_deferred_temp_storage",
    )


def test_scoped_segmentation_adapters_advertise_deferred_storage():
    pytest.importorskip("cutlass")
    from cuda.coop.cutlass._dsl.block import _difference, _discontinuity

    assert _difference._adjacent_difference_subtract_left_provider._supports_deferred_temp_storage
    assert _difference._adjacent_difference_subtract_right_provider._supports_deferred_temp_storage
    assert _discontinuity._discontinuity_flag_heads_provider._supports_deferred_temp_storage
    assert _discontinuity._discontinuity_flag_tails_provider._supports_deferred_temp_storage
    assert _discontinuity._discontinuity_flag_heads_and_tails_provider._supports_deferred_temp_storage


@pytest.mark.parametrize(
    "request_factory",
    [_adjacent_difference_request, _discontinuity_request],
)
def test_segmentation_requests_preserve_root_and_scoped_provenance(request_factory):
    root = request_factory(external_scratch=True, source="cutlass_root")
    scoped = request_factory(external_scratch=True, source="scoped_block")

    assert root.plan.call.source == "cutlass_root"
    assert scoped.plan.call.source == "scoped_block"
    assert root.plan.artifact_key == scoped.plan.artifact_key


def test_root_radix_and_merge_sort_forward_temp_storage(
    monkeypatch, set_cutlass_launch_facts
):
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass")
    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as merge_provider
    from cuda.coop.cutlass._dsl import _cub_radix_provider as radix_provider

    calls = []
    monkeypatch.setattr(
        radix_provider,
        "provider_radix_sort_keys",
        lambda **kwargs: calls.append(("radix-keys", kwargs)) or "radix-result",
    )
    monkeypatch.setattr(
        radix_provider,
        "provider_radix_sort_pairs",
        lambda **kwargs: calls.append(("radix-pairs", kwargs)) or "radix-pairs-result",
    )
    monkeypatch.setattr(
        merge_provider,
        "provider_merge_sort",
        lambda **kwargs: (
            calls.append(
                (
                    "merge-pairs" if kwargs["values"] is not None else "merge-keys",
                    kwargs,
                )
            )
            or "merge-result"
        ),
    )
    storage = object()
    keys = object()

    assert (
        coop.radix_sort_keys(
            coop.this_block(),
            keys,
            temp_storage=storage,
        )
        == "radix-result"
    )
    assert calls[-1][0] == "radix-keys"
    assert calls[-1][1]["temp_storage"] is storage

    assert (
        coop.radix_sort_pairs(
            coop.this_block(),
            keys,
            object(),
            temp_storage=storage,
        )
        == "radix-pairs-result"
    )
    assert calls[-1][0] == "radix-pairs"
    assert calls[-1][1]["temp_storage"] is storage

    assert (
        coop.merge_sort_keys(
            coop.this_block(),
            keys,
            temp_storage=storage,
        )
        == "merge-result"
    )
    assert calls[-1][0] == "merge-keys"
    assert calls[-1][1]["temp_storage"] is storage

    assert (
        coop.merge_sort_pairs(
            coop.this_block(),
            keys,
            object(),
            temp_storage=storage,
        )
        == "merge-result"
    )
    assert calls[-1][0] == "merge-pairs"
    assert calls[-1][1]["temp_storage"] is storage


def test_root_segmentation_forwards_temp_storage(monkeypatch, set_cutlass_launch_facts):
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass")
    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._dsl import (
        _cub_adjacent_difference_provider,
        _cub_discontinuity_provider,
    )

    calls = []
    monkeypatch.setattr(
        _cub_adjacent_difference_provider,
        "provider_adjacent_difference",
        lambda **kwargs: (
            calls.append(("adjacent_difference", kwargs)) or "adjacent-result"
        ),
    )
    monkeypatch.setattr(
        _cub_discontinuity_provider,
        "provider_discontinuity",
        lambda **kwargs: (
            calls.append(("discontinuity", kwargs)) or "discontinuity-result"
        ),
    )
    storage = object()
    values = object()

    assert (
        coop.adjacent_difference(
            coop.this_block(),
            values,
            temp_storage=storage,
        )
        == "adjacent-result"
    )
    assert calls[-1][0] == "adjacent_difference"
    assert calls[-1][1]["temp_storage"] is storage

    assert (
        coop.discontinuity(
            coop.this_block(),
            values,
            temp_storage=storage,
        )
        == "discontinuity-result"
    )
    assert calls[-1][0] == "discontinuity"
    assert calls[-1][1]["temp_storage"] is storage


@pytest.mark.parametrize(
    ("primitive_name", "provider_module", "provider_name"),
    [
        (
            "radix_sort_keys",
            "cuda.coop.cutlass._dsl._cub_radix_provider",
            "provider_radix_sort_keys",
        ),
        (
            "merge_sort_keys",
            "cuda.coop.cutlass._dsl._cub_merge_sort_provider",
            "provider_merge_sort",
        ),
    ],
)
def test_scoped_sort_adapters_carry_deferred_storage_in_context(
    monkeypatch,
    primitive_name,
    provider_module,
    provider_name,
):
    pytest.importorskip("cutlass")
    import importlib

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._dsl import _single_phase

    storage = coop.TempStorage()
    calls = []
    provider = importlib.import_module(provider_module)

    def capture(**kwargs):
        calls.append((kwargs, _single_phase.get_active_single_phase_context()))
        return "sorted"

    monkeypatch.setattr(provider, provider_name, capture)
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
        == "sorted"
    )
    payload, context = calls[-1]
    assert payload["temp_storage"] is None
    assert context is not None
    assert context.temp_storage is storage


@pytest.mark.parametrize(
    ("primitive_name", "provider_module", "provider_name"),
    [
        (
            "adjacent_difference_subtract_left",
            "cuda.coop.cutlass._dsl._cub_adjacent_difference_provider",
            "provider_adjacent_difference",
        ),
        (
            "adjacent_difference",
            "cuda.coop.cutlass._dsl._cub_adjacent_difference_provider",
            "provider_adjacent_difference",
        ),
        (
            "discontinuity_flag_heads",
            "cuda.coop.cutlass._dsl._cub_discontinuity_provider",
            "provider_discontinuity",
        ),
        (
            "discontinuity",
            "cuda.coop.cutlass._dsl._cub_discontinuity_provider",
            "provider_discontinuity",
        ),
    ],
)
def test_scoped_segmentation_adapters_carry_deferred_storage_in_context(
    monkeypatch,
    primitive_name,
    provider_module,
    provider_name,
):
    pytest.importorskip("cutlass")
    import importlib

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._dsl import _single_phase

    storage = coop.TempStorage()
    calls = []
    provider = importlib.import_module(provider_module)

    def capture(**kwargs):
        calls.append((kwargs, _single_phase.get_active_single_phase_context()))
        return "segmented"

    monkeypatch.setattr(provider, provider_name, capture)
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
        == "segmented"
    )
    payload, context = calls[-1]
    assert payload["temp_storage"] is None
    assert context is not None
    assert context.temp_storage is storage


def test_external_sort_and_segmentation_share_one_nvrtc_layout_bundle(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("cuda.bindings.nvrtc")
    requests = [
        _radix_request(external_scratch=True),
        _radix_pair_request(external_scratch=True),
        _merge_key_request(external_scratch=True),
        _merge_request(external_scratch=True),
        _adjacent_difference_request(external_scratch=True),
        _discontinuity_request(external_scratch=True),
    ]

    from cuda.coop.cutlass._dsl import _provider as provider_support
    from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    provider_bundle.reset_compile_state()
    probes = provider_support.bundle_scratch_layout_probes(requests)
    source = block_provider._render_bundle_source(requests)
    compilation = provider_bundle.compile_bundle_source_with_layouts(
        source,
        layout_probes=(
            provider_bundle.LayoutProbe(
                key=key,
                size_expression=probe.size_expression,
                alignment_expression=probe.alignment_expression,
            )
            for key, probe in probes.items()
        ),
        scope=block_provider._SCOPE,
        provider_dir=os.path.dirname(block_provider.__file__),
        registered_headers=block_provider._registered_cccl_headers,
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: "sm_120",
        resolve_nvrtc_arch=lambda: "compute_120",
    )

    assert compilation.path.endswith(".ltoir")
    assert os.path.getsize(compilation.path) > 0
    assert set(compilation.layouts) == set(probes)
    assert set(compilation.layouts.values()) == {
        provider_bundle.StorageLayout(size_in_bytes=512, alignment=4),
        provider_bundle.StorageLayout(size_in_bytes=516, alignment=4),
        provider_bundle.StorageLayout(size_in_bytes=1032, alignment=8),
        provider_bundle.StorageLayout(size_in_bytes=2336, alignment=16),
    }
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1


def test_external_radix_restores_provider_session_after_ffi_failure(monkeypatch):
    pytest.importorskip("cutlass")
    from cutlass._mlir import ir
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._dsl import _cub_radix_provider as radix_provider
    from cuda.coop.cutlass._dsl import _provider as provider_support

    class FakeTensor:
        iterator = SimpleNamespace(llvm_ptr=object())

        def __getitem__(self, _index):
            return object()

    storage = coop.TempStorage()
    keys = coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    snapshot = object()
    restored = []
    registrations = []
    monkeypatch.setattr(
        radix_provider._cute,
        "make_rmem_tensor",
        lambda *_args: FakeTensor(),
    )
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
        lambda _request: registrations.append("request"),
    )
    monkeypatch.setattr(
        provider_support,
        "register_deferred_temp_storage_event",
        lambda *_args, **_kwargs: (
            registrations.append("event") or (object(), object(), object())
        ),
    )

    def failing_ffi(**_kwargs):
        def invoke(*_args):
            raise RuntimeError("forced FFI failure")

        return invoke

    monkeypatch.setattr(radix_provider, "ffi", failing_ffi)
    with pytest.raises(RuntimeError, match="forced FFI failure"):
        with ir.Context():
            radix_provider.provider_radix_sort_keys(
                group=coop.this_block(),
                launch=LaunchFacts(exact_block_dim=64),
                keys=keys,
                begin_bit=0,
                end_bit=32,
                descending=False,
                source="cutlass_root",
                temp_storage=storage,
            )

    assert registrations == ["request", "event"]
    assert restored == [snapshot]


@pytest.mark.parametrize("primitive_name", ("adjacent_difference", "discontinuity"))
def test_external_segmentation_restores_provider_session_after_ffi_failure(
    monkeypatch,
    primitive_name,
):
    pytest.importorskip("cutlass")
    from cutlass._mlir import ir
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import (
        BlockAdjacentDifferenceDirection,
        BlockDiscontinuityMode,
    )
    from cuda.coop.cutlass._dsl import _provider as provider_support

    if primitive_name == "adjacent_difference":
        from cuda.coop.cutlass._dsl import (
            _cub_adjacent_difference_provider as provider,
        )
    else:
        from cuda.coop.cutlass._dsl import _cub_discontinuity_provider as provider

    class FakeTensor:
        iterator = SimpleNamespace(llvm_ptr=object())

        def __getitem__(self, _index):
            return object()

    storage = coop.TempStorage()
    values = coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    snapshot = object()
    restored = []
    registrations = []
    monkeypatch.setattr(provider._cute, "make_rmem_tensor", lambda *_args: FakeTensor())
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
        lambda _request: registrations.append("request"),
    )
    monkeypatch.setattr(
        provider_support,
        "register_deferred_temp_storage_event",
        lambda *_args, **_kwargs: (
            registrations.append("event") or (object(), object(), object())
        ),
    )

    def failing_ffi(**_kwargs):
        def invoke(*_args):
            raise RuntimeError("forced FFI failure")

        return invoke

    monkeypatch.setattr(provider, "ffi", failing_ffi)
    with pytest.raises(RuntimeError, match="forced FFI failure"):
        with ir.Context():
            if primitive_name == "adjacent_difference":
                provider.provider_adjacent_difference(
                    group=coop.this_block(),
                    launch=LaunchFacts(exact_block_dim=64),
                    value=values,
                    direction=BlockAdjacentDifferenceDirection.LEFT,
                    source="cutlass_root",
                    temp_storage=storage,
                )
            else:
                provider.provider_discontinuity(
                    group=coop.this_block(),
                    launch=LaunchFacts(exact_block_dim=64),
                    value=values,
                    mode=BlockDiscontinuityMode.HEADS,
                    source="cutlass_root",
                    temp_storage=storage,
                )

    assert registrations == ["request", "event"]
    assert restored == [snapshot]

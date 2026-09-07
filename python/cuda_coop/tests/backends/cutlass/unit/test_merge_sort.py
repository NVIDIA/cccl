# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def _provider_dependencies() -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")


def _launch_facts(block_dim: tuple[int, int, int] = (64, 1, 1)):
    from cuda.coop._core import LaunchFactOrigin, LaunchFacts

    return LaunchFacts(
        exact_block_dim=block_dim,
        provenance=LaunchFactOrigin(
            "exact_block_dim",
            "test_kernel",
            verified=True,
        ),
    )


def test_public_merge_sort_exports_and_signatures() -> None:
    _provider_dependencies()

    import cuda.coop.cutlass as coop

    for name in ("merge_sort_keys", "merge_sort_pairs"):
        assert name in coop.__all__
        function = getattr(coop, name)
        assert function.__module__ == "cuda.coop.cutlass._group_merge_sort"
        assert all(
            not parameter.startswith("_")
            for parameter in inspect.signature(function).parameters
        )


def test_frontends_delegate_resolved_block_and_warp_groups(monkeypatch) -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import api as portable_api
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import _merge_sort as provider

    monkeypatch.setattr(_launch, "current_kernel_launch_facts", _launch_facts)
    keys = cutlass_coop.ThreadData.from_values(
        Int32(3),
        Int32(1),
        dtype=Int32,
    )
    values = cutlass_coop.ThreadData.from_values(
        Int32(30),
        Int32(10),
        dtype=Int32,
    )
    calls: list[dict[str, object]] = []

    def capture(**payload):
        calls.append(payload)
        return (
            (payload["keys"], payload["values"])
            if payload["values"] is not None
            else payload["keys"]
        )

    monkeypatch.setattr(provider, "provider_merge_sort", capture)

    qualified = cutlass_coop.merge_sort_pairs(
        cutlass_coop.this_block(),
        keys,
        values,
        descending=True,
    )
    with portable_api._compiler_scope("cuda.coop.cutlass"):
        common = coop.merge_sort_keys(
            coop.this_warp(),
            keys,
            valid_items=61,
            oob_default=Int32(2_147_483_647),
        )

    assert qualified == (keys, values)
    assert common is keys
    assert [call["group"].kind for call in calls] == ["block", "warp"]
    assert all(call["group"].hierarchy.block_dim == (64, 1, 1) for call in calls)
    assert calls[0]["descending"] is True
    assert calls[1]["valid_items"] == 61
    assert calls[1]["oob_default"] == Int32(2_147_483_647)
    assert all(call["source"] == "cutlass_root" for call in calls)


def test_merge_sort_plans_render_block_external_and_warp_internal_storage() -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._lowering import _merge_sort as provider
    from cuda.coop.cutlass._lowering._core import render_cutlass_core_artifact

    block = provider._make_request(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=2,
        descending=True,
        valid_items=117,
        oob_default=Int32(-2_147_483_648),
        source="cutlass_root",
        external_scratch=True,
    )
    block_source = "\n".join(render_cutlass_core_artifact(block))
    assert block.external_scratch is True
    assert block.symbol_name.endswith("_partial_external_scratch")
    assert "::cub::BlockMergeSort<int, 64, 2, ::cub::NullType, 1, 1>" in block_source
    assert "unsigned int temp_storage_smem_addr" in block_source
    assert "temp_storage_bytes < required_temp_bytes" in block_source
    assert "required_temp_alignment - 1ull" in block_source
    assert "cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr)" in block_source
    assert "if (temp_storage_auto_sync != 0)" in block_source
    assert "if (valid_items < 0 || valid_items > 128)" in block_source

    logical_warp = provider._make_request(
        group=coop.this_warp().group_by(8),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=2,
        descending=False,
        valid_items=None,
        oob_default=None,
        source="cutlass_root",
    )
    warp_source = "\n".join(render_cutlass_core_artifact(logical_warp))
    assert logical_warp.external_scratch is False
    assert "::cub::WarpMergeSort<int, 2, 8, ::cub::NullType>" in warp_source
    assert "TempStorage storage[8]" in warp_source
    assert "cuda_coop_cutlass_linear_tid() / 8u" in warp_source
    assert "temp_storage_smem_addr" not in warp_source


@pytest.mark.parametrize("key_type_name", ["Float32", "Float64"])
def test_qualified_block_accepts_floating_point_keys(key_type_name) -> None:
    _provider_dependencies()
    from cutlass.base_dsl import typing

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._lowering import _merge_sort as provider

    key_type = getattr(typing, key_type_name)
    keys = coop.ThreadData.from_values(key_type(3), dtype=key_type)

    resolved_type, values, *_ = provider._resolve_inputs(
        group=coop.this_block(),
        keys=keys,
        values=None,
    )

    assert resolved_type is key_type
    assert len(values) == 1


def test_valid_items_and_sentinel_validation_is_strict_and_lossless() -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._lowering import _merge_sort as provider

    request = provider._make_request(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=2,
        descending=False,
        valid_items=None,
        oob_default=None,
        source="cutlass_root",
    )
    group = request.plan.resolved_group

    for invalid in (True, np.bool_(True)):
        with pytest.raises(TypeError, match="integer, not bool"):
            provider._validate_valid_items(
                invalid,
                group=group,
                items_per_thread=2,
            )
    for invalid in (1.5, np.float32(1.5), "17"):
        with pytest.raises(TypeError, match="must be an integer"):
            provider._validate_valid_items(
                invalid,
                group=group,
                items_per_thread=2,
            )
    assert isinstance(
        provider._validate_valid_items(
            np.int64(17),
            group=group,
            items_per_thread=2,
        ),
        Int32,
    )
    for invalid in (-1, 129):
        with pytest.raises(ValueError, match=r"\[0, 128\]"):
            provider._validate_valid_items(
                invalid,
                group=group,
                items_per_thread=2,
            )

    with pytest.raises(ValueError, match="not representable"):
        provider._coerce_oob_default(1 << 31, Int32)
    with pytest.raises(TypeError, match="dtype does not match"):
        provider._coerce_oob_default(1.0, Int32)
    with pytest.raises(TypeError, match="must match the key dtype"):
        provider._coerce_oob_default(np.int64(7), Int32)
    assert isinstance(provider._coerce_oob_default(np.int32(7), Int32), Int32)


def test_external_storage_is_block_only_and_forwards_fixed_binding(
    monkeypatch,
) -> None:
    _provider_dependencies()

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._lowering import _merge_sort as provider

    storage = coop.TempStorage(4096, alignment=32, auto_sync=False)
    assert (
        provider._temp_storage_for_merge_sort(
            group=coop.this_block(),
            source="cutlass_root",
            explicit_temp_storage=storage,
        )
        is storage
    )
    with pytest.raises(ValueError, match="only for block groups"):
        provider._temp_storage_for_merge_sort(
            group=coop.this_warp(),
            source="cutlass_root",
            explicit_temp_storage=storage,
        )
    with pytest.raises(ValueError, match="group-first call"):
        provider._temp_storage_for_merge_sort(
            group=coop.this_block(),
            source="legacy",
            explicit_temp_storage=storage,
        )
    exclusive = coop.TempStorage(
        4096,
        alignment=16,
        auto_sync=False,
        sharing="exclusive",
    )
    with pytest.raises(ValueError, match="does not support sharing='exclusive'"):
        provider._temp_storage_for_merge_sort(
            group=coop.this_block(),
            source="cutlass_root",
            explicit_temp_storage=exclusive,
        )

    address = object()
    observed: dict[str, object] = {}

    def materialize(temp_storage, **kwargs):
        observed["storage"] = temp_storage
        observed.update(kwargs)
        return SimpleNamespace(
            smem_addr_u32=address,
            size_in_bytes=4096,
            alignment=32,
            auto_sync=False,
        )

    monkeypatch.setattr(
        provider._provider_storage,
        "materialize_temp_storage_binding",
        materialize,
    )
    monkeypatch.setattr(provider._provider_types, "Int32", lambda value: value)
    assert provider._external_scratch_args(
        storage,
        requirement_key=("merge-sort",),
    ) == (address, 4096, 0)
    assert observed == {
        "storage": storage,
        "scope": "cuda.coop.cutlass",
        "implicit_alignment": 16,
    }


def test_external_storage_failure_rolls_back_provider_session(monkeypatch) -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._lowering import _merge_sort as provider

    request_plan = provider._make_request(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=1,
        descending=False,
        valid_items=None,
        oob_default=None,
        source="cutlass_root",
        external_scratch=True,
    )
    group = request_plan.plan.resolved_group
    keys = coop.ThreadData.from_values(Int32(3), dtype=Int32)
    storage = coop.TempStorage(4096, alignment=16)
    fake_result = SimpleNamespace(iterator=SimpleNamespace(llvm_ptr=object()))
    fake_request = SimpleNamespace(
        plan=SimpleNamespace(resolved_group=group),
        scratch_requirement_key=("merge-sort",),
        symbol_name="cuda_coop_test_merge_sort",
        ffi_param_types=(),
        bind_ffi_arguments=lambda *_args, **_kwargs: (),
    )
    events: list[str] = []

    monkeypatch.setattr(
        provider,
        "_resolve_inputs",
        lambda **_kwargs: (Int32, (Int32(3),), keys, None, None, None),
    )
    monkeypatch.setattr(provider, "_make_request", lambda **_kwargs: fake_request)
    monkeypatch.setattr(provider, "validate_operand_domains", lambda *_a, **_k: None)
    monkeypatch.setattr(
        provider._cute,
        "make_rmem_tensor",
        lambda *_args, **_kwargs: fake_result,
    )
    monkeypatch.setattr(
        provider._provider_state,
        "snapshot_active_session_state",
        lambda: events.append("snapshot") or "snapshot",
    )
    monkeypatch.setattr(
        provider._provider_state,
        "restore_active_session_state",
        lambda snapshot: events.append(f"restore:{snapshot}"),
    )
    monkeypatch.setattr(
        provider._provider_state,
        "register_request",
        lambda _request: events.append("register"),
    )
    monkeypatch.setattr(
        provider,
        "_external_scratch_args",
        lambda *_args, **_kwargs: events.append("scratch") or (1, 4096, 1),
    )

    def fail_ffi(**_kwargs):
        def invoke(*_args):
            raise RuntimeError("injected FFI failure")

        return invoke

    monkeypatch.setattr(provider, "ffi", fail_ffi)

    with pytest.raises(RuntimeError, match="injected FFI failure"):
        provider.provider_merge_sort(
            group=group,
            launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
            keys=keys,
            values=None,
            descending=False,
            source="cutlass_root",
            temp_storage=storage,
        )

    assert events == ["snapshot", "register", "scratch", "restore:snapshot"]

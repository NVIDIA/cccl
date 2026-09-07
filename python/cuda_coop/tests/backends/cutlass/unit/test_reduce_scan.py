# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


def _provider_dependencies() -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")


def _launch_facts(block_dim=64):
    from cuda.coop._core import LaunchFactOrigin, LaunchFacts

    return LaunchFacts(
        exact_block_dim=block_dim,
        exact_grid_dim=(2, 1, 1),
        exact_cluster_dim=(2, 1, 1),
        cluster_launch=True,
        provenance=(LaunchFactOrigin("cluster_launch", "test_kernel", verified=True),),
    )


def test_public_reduce_scan_exports_and_frontends(monkeypatch) -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Boolean, Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import BlockReduceAlgorithm
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import _reduce as reduce_provider
    from cuda.coop.cutlass._lowering import _scan as scan_provider

    monkeypatch.setattr(_launch, "current_kernel_launch_facts", _launch_facts)
    reduce_calls = []
    scan_calls = []
    monkeypatch.setattr(
        reduce_provider,
        "provider_reduce",
        lambda **payload: reduce_calls.append(payload) or "reduced",
    )
    monkeypatch.setattr(
        scan_provider,
        "provider_scan",
        lambda **payload: scan_calls.append(payload) or "scanned",
    )

    block = coop.this_block()
    assert (
        coop.sum(
            block,
            Int32(1),
            broadcast=False,
            valid_items=47,
            algorithm="raking",
        )
        == "reduced"
    )
    assert reduce_calls[-1]["valid_items_binding"] == ArgumentBinding.static(47)
    assert reduce_calls[-1]["algorithm"] is BlockReduceAlgorithm.RAKING
    assert (
        coop.exclusive_scan(
            block,
            Int32(1),
            scan_op="max",
            initial_value=-2_147_483_648,
            temp_storage=object(),
        )
        == "scanned"
    )
    assert scan_calls[-1]["mode"] == "exclusive"
    assert scan_calls[-1]["op"] == "max"

    expected_modules = {
        "reduce": "cuda.coop.cutlass._group_reduce",
        "sum": "cuda.coop.cutlass._group_reduce",
        "scan": "cuda.coop.cutlass._group_scan",
        "exclusive_sum": "cuda.coop.cutlass._group_scan",
        "inclusive_sum": "cuda.coop.cutlass._group_scan",
        "exclusive_scan": "cuda.coop.cutlass._group_scan",
        "inclusive_scan": "cuda.coop.cutlass._group_scan",
    }
    for name, module in expected_modules.items():
        function = getattr(coop, name)
        assert name in coop.__all__
        assert function.__module__ == module
        assert all(
            not parameter.startswith("_")
            for parameter in inspect.signature(function).parameters
        )

    for invalid in (True, __import__("numpy").bool_(True), Boolean(True)):
        with pytest.raises(TypeError, match="must be an integer"):
            coop.sum(block, Int32(1), broadcast=False, valid_items=invalid)
        with pytest.raises(TypeError, match="must be an integer"):
            coop.scan(coop.this_warp(), Int32(1), valid_items=invalid)
        with pytest.raises(TypeError, match="algorithm must not be boolean"):
            coop.sum(
                block,
                Int32(1),
                broadcast=False,
                algorithm=invalid,
            )
        with pytest.raises(TypeError, match="algorithm must not be boolean"):
            coop.scan(block, Int32(1), algorithm=invalid)
    for invalid in (1.5, object()):
        with pytest.raises(TypeError, match="must be an integer"):
            coop.sum(block, Int32(1), broadcast=False, valid_items=invalid)
        with pytest.raises(TypeError, match="must be an integer"):
            coop.scan(coop.this_warp(), Int32(1), valid_items=invalid)
    with pytest.raises(TypeError, match="broadcast must be a bool"):
        coop.sum(block, Int32(1), broadcast=1)
    with pytest.raises(NotImplementedError, match="does not support grid"):
        coop.sum(coop.this_grid(), Int32(1))


def test_reduce_plans_cover_cudax_mappings_and_direct_cub() -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import (
        ArgumentBinding,
        GroupLoweringTarget,
        ReduceValueKind,
    )
    from cuda.coop._core.block import BlockReduceAlgorithm
    from cuda.coop.cutlass._lowering import _reduce as provider

    launch = _launch_facts()
    groups = (
        coop.this_thread(),
        coop.this_warp(),
        coop.this_warp().group_by(8),
        coop.this_block().group_by(1),
        coop.this_block(),
        coop.this_cluster(),
    )
    plans = []
    for group in groups:
        plan = provider._make_group_reduce_plan(
            group=group,
            launch=launch,
            dtype=Int32,
            value_kind=ReduceValueKind.SCALAR,
            items_per_thread=1,
            op="sum",
            broadcast=True,
        ).require_supported()
        assert plan.target is GroupLoweringTarget.CUDAX_GROUP
        plans.append(plan)

    for plan, kind in zip(plans[:2], ("thread", "warp"), strict=True):
        physical_request = provider._CudaxReduceRequest(
            plan=plan,
            op="sum",
            value_type=Int32,
        )
        physical_source = "\n".join(provider._render_cudax_reduce(physical_request))
        assert "group{}" not in physical_source
        hierarchy = (
            "::cuda::experimental::implicit_hierarchy()"
            if plan.resolved_group.hierarchy.implicit
            else "hierarchy"
        )
        assert (
            f"::cuda::experimental::this_{kind} group{{{hierarchy}}};"
        ) in physical_source

    mapped_request = provider._CudaxReduceRequest(
        plan=plans[3],
        op="sum",
        value_type=Int32,
    )
    mapped_source = "\n".join(provider._render_cudax_reduce(mapped_request))
    assert "group_by<1, true>" in mapped_source

    array_plan = provider._make_group_reduce_plan(
        group=coop.this_block(),
        launch=launch,
        dtype=Int32,
        value_kind=ReduceValueKind.ARRAY,
        items_per_thread=2,
        op="sum",
        broadcast=False,
        algorithm=BlockReduceAlgorithm.RAKING,
    ).require_supported()
    array_request = provider._CubReduceRequest(array_plan, "sum", Int32)
    array_source = "\n".join(provider._render_cub_reduce(array_request))
    assert array_plan.target is GroupLoweringTarget.CUB_BLOCK
    assert "::cub::BlockReduce<int, 64" in array_source
    assert ".Sum(thread_data)" in array_source

    def logical_request(width: int):
        plan = provider._make_group_reduce_plan(
            group=coop.this_warp().group_by(width),
            launch=launch,
            dtype=Int32,
            value_kind=ReduceValueKind.SCALAR,
            items_per_thread=1,
            op="sum",
            broadcast=False,
            valid_items=ArgumentBinding.static(width - 1),
        ).require_supported()
        assert plan.target is GroupLoweringTarget.CUB_WARP
        return provider._CubReduceRequest(plan, "sum", Int32)

    logical_8 = logical_request(8)
    logical_16 = logical_request(16)
    assert logical_8.symbol_name != logical_16.symbol_name
    logical_source = "\n".join(provider._render_cub_reduce(logical_8))
    assert "::cub::WarpReduce<int, 8>" in logical_source
    assert "TempStorage storage[8]" in logical_source
    assert ".Sum(item0, 7)" in logical_source


def test_scan_plans_render_aggregate_storage_and_distinct_widths(monkeypatch) -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Float32, Float64, Int32, Uint8

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, ScanValueKind
    from cuda.coop.cutlass._compiler import _storage as provider_storage
    from cuda.coop.cutlass._compiler import _types as provider_types
    from cuda.coop.cutlass._lowering import _scan as provider

    launch = _launch_facts()

    def logical_request(width: int):
        plan = provider._make_group_scan_plan(
            group=coop.this_warp().group_by(width),
            launch=launch,
            dtype=Int32,
            value_kind=ScanValueKind.SCALAR,
            items_per_thread=1,
            mode="inclusive",
            op="sum",
            aggregate=True,
            valid_items=width - 1,
        ).require_supported()
        assert plan.target is GroupLoweringTarget.CUB_WARP
        return provider._CubScanRequest(plan, "sum", Int32)

    logical_8 = logical_request(8)
    logical_16 = logical_request(16)
    assert logical_8.symbol_name != logical_16.symbol_name
    logical_source = "\n".join(provider._render_cub_scan(logical_8))
    assert "::cub::WarpScan<int, 8>" in logical_source
    assert "TempStorage storage[8]" in logical_source
    assert (
        ".InclusiveScanPartial(value, result, ::cuda::std::plus<>{}, 7, aggregate)"
        in logical_source
    )

    for value_type, zero_literal in (
        (Uint8, "0u"),
        (Float32, "0.0f"),
        (Float64, "0.0"),
    ):
        partial_exclusive_plan = provider._make_group_scan_plan(
            group=coop.this_warp().group_by(8),
            launch=launch,
            dtype=value_type,
            value_kind=ScanValueKind.SCALAR,
            items_per_thread=1,
            mode="exclusive",
            op="sum",
            valid_items=5,
        ).require_supported()
        partial_exclusive_request = provider._CubScanRequest(
            partial_exclusive_plan,
            "sum",
            value_type,
        )
        partial_exclusive_source = "\n".join(
            provider._render_cub_scan(partial_exclusive_request)
        )
        assert (
            ".ExclusiveScanPartial(value, result, "
            f"{zero_literal}, ::cuda::std::plus<>{{}}, 5)" in partial_exclusive_source
        )

    for feature in ("reduce", "scan"):
        for op in ("bit_and", "bit_or", "bit_xor"):
            provider_types.validate_scan_reduce_op_for_type(
                op,
                Uint8,
                root_scope="cuda.coop.cutlass",
                feature=feature,
                namespace="thread_group",
            )
            with pytest.raises(TypeError, match="require an integral type"):
                provider_types.validate_scan_reduce_op_for_type(
                    op,
                    Float32,
                    root_scope="cuda.coop.cutlass",
                    feature=feature,
                    namespace="thread_group",
                )

    block_plan = provider._make_group_scan_plan(
        group=coop.this_block(),
        launch=launch,
        dtype=Int32,
        value_kind=ScanValueKind.ARRAY,
        items_per_thread=2,
        mode="exclusive",
        op="sum",
        aggregate=True,
    ).require_supported()
    external_plan = provider._with_caller_owned_scan_storage(block_plan)
    external_request = provider._CubScanRequest(
        external_plan,
        "sum",
        Int32,
        external_scratch=True,
    )
    external_source = "\n".join(provider._render_cub_scan(external_request))
    assert "reinterpret_cast" in external_source
    assert "temp_storage_bytes < required_temp_bytes" in external_source
    assert "temp_storage_auto_sync != 0" in external_source
    assert "__shared__ typename implementation_type::TempStorage" not in (
        external_source
    )
    probe = provider._cub_scan_scratch_layout_probe(external_request)
    assert probe is not None
    assert probe.requirement_key == external_request.scratch_requirement_key

    fixed = coop.TempStorage(4096, alignment=16, auto_sync=False)
    assert (
        provider._temp_storage_for_scan(
            group=coop.this_block(),
            explicit_temp_storage=fixed,
        )
        is fixed
    )
    with pytest.raises(ValueError, match="sharing='exclusive'"):
        provider._temp_storage_for_scan(
            group=coop.this_block(),
            explicit_temp_storage=coop.TempStorage(4096, sharing="exclusive"),
        )
    with pytest.raises(ValueError, match="only for block groups"):
        provider._temp_storage_for_scan(
            group=coop.this_warp(),
            explicit_temp_storage=fixed,
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
        fixed,
        requirement_key=external_request.scratch_requirement_key,
    ) == ("shared-address", Int32(4096), Int32(0))


class _FakeTensor:
    class _Iterator:
        llvm_ptr = object()

    iterator = _Iterator()

    def __getitem__(self, index):
        return index


def test_fixed_scan_failure_rolls_back_provider_session(monkeypatch) -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import ScanValueKind
    from cuda.coop.cutlass._compiler import _state as provider_state
    from cuda.coop.cutlass._lowering import _scan as provider

    plan = provider._make_group_scan_plan(
        group=coop.this_block(),
        launch=_launch_facts(),
        dtype=Int32,
        value_kind=ScanValueKind.ARRAY,
        items_per_thread=2,
        mode="inclusive",
        op="sum",
    ).require_supported()
    plan = provider._with_caller_owned_scan_storage(plan)
    value = coop.ThreadData.from_values(Int32(1), Int32(2), dtype=Int32)
    storage = coop.TempStorage(4096, alignment=16)
    snapshot = object()
    restored = []
    monkeypatch.setattr(
        provider_state,
        "snapshot_active_session_state",
        lambda: snapshot,
    )
    monkeypatch.setattr(
        provider_state,
        "restore_active_session_state",
        restored.append,
    )
    monkeypatch.setattr(provider_state, "register_request", lambda _request: None)
    monkeypatch.setattr(
        provider,
        "_external_scratch_args",
        lambda *_args, **_kwargs: (object(), object(), object()),
    )
    monkeypatch.setattr(
        provider._cute, "make_rmem_tensor", lambda *_args: _FakeTensor()
    )
    monkeypatch.setattr(provider.llvm.PointerType, "get", lambda *_args: object())

    def failing_ffi(**_kwargs):
        def invoke(*_args):
            raise RuntimeError("forced FFI failure")

        return invoke

    monkeypatch.setattr(provider, "ffi", failing_ffi)
    with pytest.raises(RuntimeError, match="forced FFI failure"):
        provider._materialize_scan(
            plan=plan,
            value=value,
            values=(Int32(1), Int32(2)),
            value_type=Int32,
            op="sum",
            initial_value=None,
            aggregate_output=None,
            valid_items=None,
            external_temp_storage=storage,
        )

    assert restored == [snapshot]

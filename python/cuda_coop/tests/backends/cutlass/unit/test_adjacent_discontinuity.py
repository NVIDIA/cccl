# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _launch_facts(block_dim=64):
    from cuda.coop._core import LaunchFactOrigin, LaunchFacts

    return LaunchFacts(
        exact_block_dim=block_dim,
        exact_grid_dim=(2, 1, 1),
        exact_cluster_dim=(2, 1, 1),
        cluster_launch=True,
        provenance=(LaunchFactOrigin("cluster_launch", "test_kernel", verified=True),),
    )


def test_group_comparison_calls_route_to_exact_cub_block_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import (
        _adjacent_difference as adjacent_provider,
    )
    from cuda.coop.cutlass._lowering import (
        _discontinuity as discontinuity_provider,
    )

    monkeypatch.setattr(
        _launch,
        "current_kernel_launch_facts",
        lambda: _launch_facts((8, 4, 2)),
    )

    items = coop.ThreadData.from_values(2, 2, dtype=Int32)
    storage = coop.TempStorage()
    adjacent_calls: list[dict[str, object]] = []
    discontinuity_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        adjacent_provider,
        "provider_adjacent_difference",
        lambda **kwargs: adjacent_calls.append(kwargs) or "difference",
    )
    monkeypatch.setattr(
        discontinuity_provider,
        "provider_discontinuity",
        lambda **kwargs: discontinuity_calls.append(kwargs) or "flags",
    )

    block = coop.this_block()
    assert (
        coop.adjacent_difference(
            block,
            items,
            valid_items=125,
            tile_predecessor_item=-7,
            temp_storage=storage,
        )
        == "difference"
    )
    assert (
        coop.adjacent_difference(
            block,
            items,
            direction="right",
            tile_successor_item=211,
            temp_storage=storage,
        )
        == "difference"
    )
    assert (
        coop.discontinuity(
            block,
            items,
            tile_predecessor_item=-7,
            temp_storage=storage,
        )
        == "flags"
    )
    assert (
        coop.discontinuity(
            block,
            items,
            mode="tails",
            tile_successor_item=211,
            temp_storage=storage,
        )
        == "flags"
    )

    assert [call["direction"].value for call in adjacent_calls] == [
        "left",
        "right",
    ]
    assert adjacent_calls[0]["valid_items"] == 125
    assert adjacent_calls[0]["tile_predecessor_item"] == -7
    assert adjacent_calls[1]["tile_successor_item"] == 211
    assert [call["mode"].value for call in discontinuity_calls] == [
        "heads",
        "tails",
    ]
    assert discontinuity_calls[0]["tile_predecessor_item"] == -7
    assert discontinuity_calls[1]["tile_successor_item"] == 211
    for call in (*adjacent_calls, *discontinuity_calls):
        assert call["source"] == "cutlass_root"
        assert call["temp_storage"] is storage

    launch = LaunchFacts(exact_block_dim=(8, 4, 2))
    left_plan = adjacent_provider._make_group_adjacent_difference_plan(
        group=block,
        launch=launch,
        dtype=Int32,
        items_per_thread=2,
        direction="left",
        valid_items=125,
        tile_predecessor_item=-7,
    ).require_supported()
    right_plan = adjacent_provider._make_group_adjacent_difference_plan(
        group=block,
        launch=launch,
        dtype=Int32,
        items_per_thread=2,
        direction="right",
        tile_successor_item=211,
    ).require_supported()
    heads_plan = discontinuity_provider._make_group_discontinuity_plan(
        group=block,
        launch=launch,
        dtype=Int32,
        flag_dtype=Int32,
        items_per_thread=2,
        mode="heads",
        tile_predecessor_item=-7,
    ).require_supported()
    tails_plan = discontinuity_provider._make_group_discontinuity_plan(
        group=block,
        launch=launch,
        dtype=Int32,
        flag_dtype=Int32,
        items_per_thread=2,
        mode="tails",
        tile_successor_item=211,
    ).require_supported()

    assert all(
        plan.target is GroupLoweringTarget.CUB_BLOCK
        for plan in (left_plan, right_plan, heads_plan, tails_plan)
    )
    assert left_plan.implementation.method_name == "SubtractLeftPartialTile"
    assert right_plan.implementation.method_name == "SubtractRight"
    assert heads_plan.implementation.method_name == "FlagHeads"
    assert tails_plan.implementation.method_name == "FlagTails"
    assert all(
        plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 2
        for plan in (left_plan, right_plan, heads_plan, tails_plan)
    )

    for operation in (coop.adjacent_difference, coop.discontinuity):
        with pytest.raises(NotImplementedError, match="only this_block"):
            operation(
                coop.this_warp(),
                items,
            )


@pytest.mark.parametrize(
    ("operation_name", "provider_name"),
    [
        ("adjacent_difference", "provider_adjacent_difference"),
        ("discontinuity", "provider_discontinuity"),
    ],
)
def test_common_comparisons_reject_scalar_but_qualified_scalar_remains_available(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    provider_name: str,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import api as portable_api
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import (
        _adjacent_difference as adjacent_provider,
    )
    from cuda.coop.cutlass._lowering import (
        _discontinuity as discontinuity_provider,
    )

    monkeypatch.setattr(
        _launch,
        "current_kernel_launch_facts",
        lambda: _launch_facts(32),
    )

    provider = (
        adjacent_provider
        if operation_name == "adjacent_difference"
        else discontinuity_provider
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        provider_name,
        lambda **kwargs: calls.append(kwargs) or "qualified-result",
    )

    with portable_api._compiler_scope("cuda.coop.cutlass"):
        group = coop.this_block()
        with pytest.raises(
            TypeError,
            match=rf"cuda\.coop\.{operation_name} requires a fixed-size ThreadData",
        ):
            getattr(coop, operation_name)(group, 7)

    assert portable_api._common_root_operation_name() is None
    assert (
        getattr(cutlass_coop, operation_name)(
            cutlass_coop.this_block(),
            7,
        )
        == "qualified-result"
    )
    assert len(calls) == 1
    assert calls[0]["value"] == 7


@pytest.mark.parametrize(
    ("provider_module", "storage_helper", "primitive_name"),
    [
        (
            "_adjacent_difference",
            "_temp_storage_for_adjacent_difference",
            "adjacent_difference",
        ),
        (
            "_discontinuity",
            "_temp_storage_for_discontinuity",
            "discontinuity",
        ),
    ],
)
def test_fixed_comparison_storage_uses_exact_external_scratch_binding(
    monkeypatch: pytest.MonkeyPatch,
    provider_module: str,
    storage_helper: str,
    primitive_name: str,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._compiler import _storage as provider_storage
    from cuda.coop.cutlass._lowering._core import render_cutlass_core_artifact

    provider = __import__(
        f"cuda.coop.cutlass._lowering.{provider_module}",
        fromlist=[provider_module],
    )
    storage = coop.TempStorage(4096, alignment=64, auto_sync=False)
    assert (
        getattr(provider, storage_helper)(
            group=coop.this_block(),
            explicit_temp_storage=storage,
        )
        is storage
    )
    with pytest.raises(ValueError, match="sharing='exclusive'"):
        getattr(provider, storage_helper)(
            group=coop.this_block(),
            explicit_temp_storage=coop.TempStorage(4096, sharing="exclusive"),
        )
    with pytest.raises(ValueError, match="only for block groups"):
        getattr(provider, storage_helper)(
            group=coop.this_warp(),
            explicit_temp_storage=storage,
        )

    materialized = []

    def bind_storage(bound_storage, **kwargs):
        materialized.append((bound_storage, kwargs))
        return SimpleNamespace(
            smem_addr_u32="shared-address",
            size_in_bytes=4096,
            alignment=64,
            auto_sync=False,
        )

    monkeypatch.setattr(
        provider_storage,
        "materialize_temp_storage_binding",
        bind_storage,
    )
    requirement_key = (primitive_name, "i32", 2)
    assert provider._external_scratch_args(
        storage,
        requirement_key=requirement_key,
    ) == ("shared-address", Int32(4096), Int32(0))
    assert materialized == [
        (
            storage,
            {
                "scope": "cuda.coop.cutlass",
                "implicit_alignment": 16,
            },
        )
    ]

    request_kwargs = {
        "group": coop.this_block(),
        "launch": _launch_facts(),
        "value_type": Int32,
        "items_per_thread": 2,
        "source": "cutlass_root",
        "external_scratch": True,
    }
    if primitive_name == "adjacent_difference":
        from cuda.coop._core.block import BlockAdjacentDifferenceDirection

        request = provider._make_request(
            **request_kwargs,
            direction=BlockAdjacentDifferenceDirection.LEFT,
            valid_items=125,
            tile_predecessor_item=Int32(-13),
            tile_successor_item=None,
        )
    else:
        from cuda.coop._core.block import BlockDiscontinuityMode

        request = provider._make_request(
            **request_kwargs,
            mode=BlockDiscontinuityMode.HEADS_AND_TAILS,
            tile_predecessor_item=Int32(-13),
            tile_successor_item=Int32(29),
        )
    source = "\n".join(render_cutlass_core_artifact(request))
    assert "temp_storage_bytes < required_temp_bytes" in source
    assert "temp_storage_smem_addr &" in source
    assert "reinterpret_cast" in source
    assert "if (temp_storage_auto_sync != 0)" in source
    assert "__shared__ typename implementation_type::TempStorage" not in source


def test_comparison_frontends_reject_invalid_static_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import (
        _adjacent_difference as adjacent_provider,
    )
    from cuda.coop.cutlass._lowering import (
        _discontinuity as discontinuity_provider,
    )

    monkeypatch.setattr(_launch, "current_kernel_launch_facts", _launch_facts)
    monkeypatch.setattr(
        adjacent_provider,
        "provider_adjacent_difference",
        lambda **_kwargs: "difference",
    )
    monkeypatch.setattr(
        discontinuity_provider,
        "provider_discontinuity",
        lambda **_kwargs: "flags",
    )
    items = coop.ThreadData.from_values(Int32(2), Int32(3), dtype=Int32)
    block = coop.this_block()

    for invalid in (True, __import__("numpy").bool_(True), 1.5, object()):
        with pytest.raises(TypeError, match="valid_items must be an integer"):
            coop.adjacent_difference(block, items, valid_items=invalid)
    for invalid in (-1, 129):
        with pytest.raises(ValueError, match="between zero and the block tile"):
            coop.adjacent_difference(block, items, valid_items=invalid)
    with pytest.raises(NotImplementedError, match="built-in subtraction"):
        coop.adjacent_difference(block, items, difference_op=lambda x, y: x - y)
    with pytest.raises(NotImplementedError, match="built-in inequality"):
        coop.discontinuity(block, items, flag_op=lambda x, y: x != y)

    assert (
        coop.adjacent_difference(block, items, valid_items=Int32(125)) == "difference"
    )

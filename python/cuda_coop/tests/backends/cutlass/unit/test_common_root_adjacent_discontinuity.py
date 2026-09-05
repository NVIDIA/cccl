# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="cutlass", evidence="lowering"
)
@pytest.mark.evidence_for("group.discontinuity", backend="cutlass", evidence="lowering")
def test_group_comparison_calls_route_to_exact_cub_block_plans(
    monkeypatch: pytest.MonkeyPatch,
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts((8, 4, 2))
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts
    from cuda.coop.cutlass import (
        _group_adjacent_difference,
        _group_discontinuity,
    )
    from cuda.coop.cutlass._dsl import (
        _cub_adjacent_difference_provider as adjacent_provider,
    )
    from cuda.coop.cutlass._dsl import (
        _cub_discontinuity_provider as discontinuity_provider,
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
    left_plan = _group_adjacent_difference._make_group_adjacent_difference_plan(
        group=block,
        launch=launch,
        dtype=Int32,
        items_per_thread=2,
        direction="left",
        valid_items=125,
        tile_predecessor_item=-7,
    ).require_supported()
    right_plan = _group_adjacent_difference._make_group_adjacent_difference_plan(
        group=block,
        launch=launch,
        dtype=Int32,
        items_per_thread=2,
        direction="right",
        tile_successor_item=211,
    ).require_supported()
    heads_plan = _group_discontinuity._make_group_discontinuity_plan(
        group=block,
        launch=launch,
        dtype=Int32,
        flag_dtype=Int32,
        items_per_thread=2,
        mode="heads",
        tile_predecessor_item=-7,
    ).require_supported()
    tails_plan = _group_discontinuity._make_group_discontinuity_plan(
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
    set_cutlass_launch_facts,
    operation_name: str,
    provider_name: str,
) -> None:
    set_cutlass_launch_facts(32)
    pytest.importorskip("cutlass.cute.ffi")

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import root_api
    from cuda.coop.cutlass._dsl import (
        _cub_adjacent_difference_provider as adjacent_provider,
    )
    from cuda.coop.cutlass._dsl import (
        _cub_discontinuity_provider as discontinuity_provider,
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

    with root_api._compiler_scope("cuda.coop.cutlass"):
        group = coop.this_block()
        with pytest.raises(
            TypeError,
            match=rf"cuda\.coop\.{operation_name} requires a fixed-size ThreadData",
        ):
            getattr(coop, operation_name)(group, 7)

    assert root_api._common_root_operation_name() is None
    assert (
        getattr(cutlass_coop, operation_name)(
            cutlass_coop.this_block(),
            7,
        )
        == "qualified-result"
    )
    assert len(calls) == 1
    assert calls[0]["value"] == 7

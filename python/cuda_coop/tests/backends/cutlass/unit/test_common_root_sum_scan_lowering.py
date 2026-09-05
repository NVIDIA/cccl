# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest


@pytest.mark.evidence_for("group.reduce", backend="cutlass", evidence="lowering")
@pytest.mark.evidence_for("group.sum", backend="cutlass", evidence="lowering")
def test_common_reduce_sum_plans_cover_every_certified_group_route(
    monkeypatch: pytest.MonkeyPatch,
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import (
        ArgumentBinding,
        GroupLoweringTarget,
        LaunchFactOrigin,
        LaunchFacts,
        ReduceValueKind,
    )
    from cuda.coop._core.block import BlockReduceAlgorithm
    from cuda.coop.cutlass import _group_reduce
    from cuda.coop.cutlass._dsl import _cudax_reduce_provider as provider

    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_reduce",
        lambda **kwargs: observed.append(kwargs) or "summed",
    )
    assert (
        coop.sum(
            coop.this_block(),
            Int32(1),
            broadcast=False,
            valid_items=47,
            algorithm="raking",
        )
        == "summed"
    )
    assert len(observed) == 1
    call = observed[0]
    assert call["group"] == coop.this_block()
    assert call["op"] == "sum"
    assert call["broadcast"] is False
    assert call["valid_items"] == 47
    assert call["valid_items_binding"] == ArgumentBinding.static(47)
    assert call["algorithm"] is BlockReduceAlgorithm.RAKING
    assert (
        coop.reduce(
            coop.this_block(),
            Int32(1),
            binary_op="max",
            broadcast=False,
            algorithm="raking",
        )
        == "summed"
    )
    assert len(observed) == 2
    call = observed[1]
    assert call["group"] == coop.this_block()
    assert call["op"] == "max"
    assert call["broadcast"] is False
    assert call["valid_items"] is None
    assert call["algorithm"] is BlockReduceAlgorithm.RAKING

    launch = LaunchFacts(
        exact_block_dim=64,
        exact_cluster_dim=2,
        cluster_launch=True,
        provenance=(LaunchFactOrigin("cluster_launch", "test_launch", verified=True),),
    )
    groups = (
        coop.this_thread(),
        coop.this_warp(),
        coop.this_warp().group_by(8),
        coop.this_block(),
        coop.this_cluster(),
    )

    for group in groups:
        for op, broadcast in (("sum", True), ("max", False)):
            plan = _group_reduce._make_group_reduce_plan(
                group=group,
                launch=launch,
                dtype=Int32,
                value_kind=ReduceValueKind.SCALAR,
                items_per_thread=1,
                op=op,
                broadcast=broadcast,
            ).require_supported()

            assert plan.target is GroupLoweringTarget.CUDAX_GROUP
            assert plan.call.source == "cutlass_root"

    block_partial = _group_reduce._make_group_reduce_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        value_kind=ReduceValueKind.SCALAR,
        items_per_thread=1,
        op="sum",
        broadcast=False,
        valid_items=ArgumentBinding.static(47),
        algorithm=BlockReduceAlgorithm.RAKING,
    ).require_supported()
    warp_partial = _group_reduce._make_group_reduce_plan(
        group=coop.this_warp(),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        value_kind=ReduceValueKind.SCALAR,
        items_per_thread=1,
        op="sum",
        broadcast=False,
        valid_items=ArgumentBinding.static(24),
    ).require_supported()
    block_max = _group_reduce._make_group_reduce_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        value_kind=ReduceValueKind.SCALAR,
        items_per_thread=1,
        op="max",
        broadcast=False,
        algorithm=BlockReduceAlgorithm.RAKING,
    ).require_supported()
    warp_max = _group_reduce._make_group_reduce_plan(
        group=coop.this_warp(),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        value_kind=ReduceValueKind.SCALAR,
        items_per_thread=1,
        op="max",
        broadcast=False,
        valid_items=ArgumentBinding.static(24),
    ).require_supported()

    assert block_partial.target is GroupLoweringTarget.CUB_BLOCK
    assert warp_partial.target is GroupLoweringTarget.CUB_WARP
    assert block_max.target is GroupLoweringTarget.CUB_BLOCK
    assert warp_max.target is GroupLoweringTarget.CUB_WARP


@pytest.mark.evidence_for("group.scan", backend="cutlass", evidence="lowering")
@pytest.mark.evidence_for("group.exclusive_sum", backend="cutlass", evidence="lowering")
@pytest.mark.evidence_for("group.inclusive_sum", backend="cutlass", evidence="lowering")
@pytest.mark.evidence_for(
    "group.exclusive_scan", backend="cutlass", evidence="lowering"
)
@pytest.mark.evidence_for(
    "group.inclusive_scan", backend="cutlass", evidence="lowering"
)
def test_common_scan_aliases_route_exact_block_and_warp_cub_plans(
    monkeypatch: pytest.MonkeyPatch,
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts, ScanValueKind
    from cuda.coop.cutlass import _group_scan
    from cuda.coop.cutlass._dsl import _cub_scan_provider as provider

    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_scan",
        lambda **kwargs: observed.append(kwargs) or "scanned",
    )
    block = coop.this_block()
    storage = object()
    public_cases = (
        ("scan", {}, "exclusive", "sum", None),
        ("scan", {"initial_value": 7}, "exclusive", "sum", 7),
        ("scan", {"mode": "inclusive"}, "inclusive", "sum", None),
        ("exclusive_sum", {}, "exclusive", "sum", None),
        ("inclusive_sum", {}, "inclusive", "sum", None),
        (
            "exclusive_scan",
            {"scan_op": "max", "initial_value": -2_147_483_648},
            "exclusive",
            "max",
            -2_147_483_648,
        ),
        ("inclusive_scan", {"scan_op": "max"}, "inclusive", "max", None),
    )
    for name, kwargs, mode, op, initial_value in public_cases:
        result = getattr(coop, name)(
            block,
            Int32(1),
            temp_storage=storage,
            **kwargs,
        )

        assert result == "scanned"
        call = observed[-1]
        assert call["mode"] == mode
        assert call["op"] == op
        assert call["initial_value"] == initial_value
        assert call["temp_storage"] is storage
        assert call["source"] == "cutlass_root"

    cases = (
        ("exclusive", "sum", None),
        ("inclusive", "sum", None),
        ("exclusive", "max", -2_147_483_648),
        ("inclusive", "max", None),
    )

    for group, expected_target in (
        (coop.this_block(), GroupLoweringTarget.CUB_BLOCK),
        (coop.this_warp(), GroupLoweringTarget.CUB_WARP),
    ):
        for mode, op, initial_value in cases:
            plan = _group_scan._make_group_scan_plan(
                group=group,
                launch=LaunchFacts(exact_block_dim=64),
                dtype=Int32,
                value_kind=ScanValueKind.SCALAR,
                items_per_thread=1,
                mode=mode,
                op=op,
                initial_value=initial_value,
            ).require_supported()

            assert plan.target is expected_target
            assert plan.call.source == "cutlass_root"


@pytest.mark.parametrize("group_name", ["this_block", "this_warp"])
@pytest.mark.parametrize("operation", ["scan", "exclusive_scan"])
def test_group_first_non_sum_exclusive_scan_requires_initial_value(
    set_cutlass_launch_facts,
    group_name: str,
    operation: str,
) -> None:
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop

    group = getattr(coop, group_name)()
    with pytest.raises(ValueError, match="requires initial_value"):
        getattr(coop, operation)(
            group,
            Int32(1),
            scan_op="max",
        )

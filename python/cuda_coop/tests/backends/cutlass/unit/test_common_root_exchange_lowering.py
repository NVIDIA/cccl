# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.mark.evidence_for("group.exchange", backend="cutlass", evidence="lowering")
def test_group_exchange_routes_both_modes_to_exact_block_and_warp_cub_plans(
    monkeypatch: pytest.MonkeyPatch,
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts
    from cuda.coop.cutlass import _group_exchange
    from cuda.coop.cutlass._dsl import _cub_exchange_provider as provider
    from cuda.coop.cutlass._limits import MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD

    assert MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD == 5
    items = coop.ThreadData.from_values(0, 1, 2, 3, 4, dtype=Int32)
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_exchange",
        lambda **kwargs: observed.append(kwargs) or "exchanged",
    )

    for group in (coop.this_block(), coop.this_warp()):
        for mode in ("striped_to_blocked", "blocked_to_striped"):
            assert (
                coop.exchange(
                    group,
                    items,
                    mode=mode,
                )
                == "exchanged"
            )
            call = observed[-1]
            assert call["group"] == group
            assert call["value"] is items
            assert call["mode"] == mode
            assert call["output"] is None
            assert call["source"] == "cutlass_root"

            plan = _group_exchange._make_group_exchange_plan(
                group=group,
                launch=LaunchFacts(exact_block_dim=64),
                dtype=Int32,
                items_per_thread=5,
                mode=mode,
            ).require_supported()
            assert plan.target is (
                GroupLoweringTarget.CUB_BLOCK
                if group.kind == "block"
                else GroupLoweringTarget.CUB_WARP
            )
            assert plan.call.source == "cutlass_root"
            assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 5
            assert plan.implementation.method_name == (
                "StripedToBlocked"
                if mode == "striped_to_blocked"
                else "BlockedToStriped"
            )

    too_many = coop.ThreadData.from_values(0, 1, 2, 3, 4, 5, dtype=Int32)
    with pytest.raises(NotImplementedError, match="at most 5 items per thread"):
        coop.exchange(
            coop.this_block(),
            too_many,
        )

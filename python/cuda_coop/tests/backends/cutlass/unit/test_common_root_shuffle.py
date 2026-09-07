# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.mark.evidence_for("group.shuffle", backend="cutlass", evidence="lowering")
def test_group_shuffle_routes_portable_modes_to_exact_cub_block_plans(
    monkeypatch: pytest.MonkeyPatch,
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts((8, 4, 2))
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts
    from cuda.coop.cutlass import _group_shuffle
    from cuda.coop.cutlass._dsl import _cub_shuffle_provider as provider

    items = coop.ThreadData.from_values(0, 1, 2, 3, dtype=Int32)
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_shuffle",
        lambda **kwargs: observed.append(kwargs) or "shuffled",
    )

    block = coop.this_block()
    for mode in ("up", "down"):
        assert (
            coop.shuffle(
                block,
                items,
                mode=mode,
                distance=1,
            )
            == "shuffled"
        )
        call = observed[-1]
        assert call["group"].block_dim == (8, 4, 2)
        assert call["value"] is items
        assert call["mode"].value == mode
        assert call["distance"] == 1
        assert call["block_prefix"] is None
        assert call["block_suffix"] is None
        assert call["source"] == "cutlass_root"

        plan = _group_shuffle._make_group_shuffle_plan(
            group=block,
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            dtype=Int32,
            items_per_thread=4,
            mode=mode,
        ).require_supported()
        assert plan.target is GroupLoweringTarget.CUB_BLOCK
        assert plan.call.source == "cutlass_root"
        assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 4
        assert plan.implementation.method_name == mode.capitalize()


@pytest.mark.evidence_for("group.shuffle", backend="cutlass", evidence="lowering")
def test_group_shuffle_rejects_nonportable_thread_data_routes(
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop

    items = coop.ThreadData.from_values(0, 1, dtype=Int32)
    with pytest.raises(NotImplementedError, match="only distance=1"):
        coop.shuffle(
            coop.this_block(),
            items,
            mode="up",
            distance=2,
        )
    with pytest.raises(NotImplementedError, match="only public-CUB Up/Down"):
        coop.shuffle(
            coop.this_block(),
            items,
            mode="rotate",
        )
    with pytest.raises(NotImplementedError, match="only this_block"):
        coop.shuffle(
            coop.this_warp(),
            items,
        )

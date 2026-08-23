# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.mark.evidence_for("group.radix_rank", backend="cutlass", evidence="lowering")
def test_common_and_qualified_radix_rank_share_exact_signed_block_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32, Uint32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import GroupOperandKind, LaunchFacts, root_api
    from cuda.coop.cutlass import _group_radix
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider
    from cuda.coop.cutlass._dsl import _launch

    launch = LaunchFacts(exact_block_dim=(8, 4, 2))
    monkeypatch.setattr(_launch, "infer_launch_facts", lambda *_args, **_kwargs: launch)

    keys = cutlass_coop.ThreadData.from_values(3, -1, dtype=Int32)
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_radix_rank",
        lambda **kwargs: observed.append(kwargs) or ("ranks", len(observed)),
    )

    qualified_result = cutlass_coop.radix_rank(
        cutlass_coop.this_block(),
        keys,
        begin_bit=28,
        end_bit=32,
        descending=True,
    )
    with root_api._compiler_scope("cuda.coop.cutlass"):
        common_result = coop.radix_rank(
            coop.this_block(),
            keys,
            begin_bit=28,
            radix_bits=4,
            descending=True,
        )

    assert qualified_result[0] == common_result[0] == "ranks"
    qualified_call, common_call = observed
    assert qualified_call["group"] == common_call["group"]
    assert qualified_call["keys"] is common_call["keys"] is keys
    assert qualified_call["begin_bit"] == common_call["begin_bit"] == 28
    assert qualified_call["end_bit"] == common_call["end_bit"] == 32
    assert qualified_call["descending"] is common_call["descending"] is True
    assert (
        qualified_call["exclusive_digit_prefix"]
        is common_call["exclusive_digit_prefix"]
        is None
    )
    assert qualified_call["source"] == common_call["source"] == "cutlass_root"

    plan = _group_radix._make_group_radix_rank_plan(
        group=common_call["group"],
        launch=launch,
        cub_key_dtype=Uint32,
        input_dtype=Int32,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        begin_bit=28,
        end_bit=32,
        key_bit_width=32,
        descending=True,
        exclusive_digit_prefix_items_per_thread=None,
        source="common_root_test",
    ).require_supported()
    assert plan.implementation.struct_name == "BlockRadixRank"
    assert plan.implementation.method_name == "RankKeys"
    assert plan.implementation.template_arguments["KeyT"] is Uint32
    assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 2
    assert plan.implementation.template_arguments["RADIX_BITS"] == 4
    assert plan.implementation.template_arguments["IS_DESCENDING"] == "true"

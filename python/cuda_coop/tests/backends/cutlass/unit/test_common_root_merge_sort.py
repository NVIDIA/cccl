# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="cutlass", evidence="lowering"
)
def test_common_and_qualified_merge_sort_share_exact_block_and_warp_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts, root_api
    from cuda.coop.cutlass import _group_merge_sort
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider
    from cuda.coop.cutlass._dsl import _launch

    launch = LaunchFacts(exact_block_dim=(8, 4, 2))
    monkeypatch.setattr(_launch, "infer_launch_facts", lambda *_args, **_kwargs: launch)

    keys = cutlass_coop.ThreadData.from_values(3, 1, 3, dtype=Int32)
    storage = cutlass_coop.TempStorage()
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_merge_sort",
        lambda **kwargs: observed.append(kwargs) or ("sorted", len(observed)),
    )
    routes = (
        ("block", False, None, None),
        ("block", True, 117, -2_147_483_648),
        ("warp", True, None, None),
        ("warp", False, 53, 2_147_483_647),
    )
    for group_kind, descending, valid_items, oob_default in routes:
        qualified_group = (
            cutlass_coop.this_block()
            if group_kind == "block"
            else cutlass_coop.this_warp()
        )
        qualified_result = cutlass_coop.merge_sort_keys(
            qualified_group,
            keys,
            descending=descending,
            valid_items=valid_items,
            oob_default=oob_default,
            temp_storage=storage if group_kind == "block" else None,
        )

        with root_api._compiler_scope("cuda.coop.cutlass"):
            common_group = (
                coop.this_block() if group_kind == "block" else coop.this_warp()
            )
            common_result = coop.merge_sort_keys(
                common_group,
                keys,
                descending=descending,
                valid_items=valid_items,
                oob_default=oob_default,
                temp_storage=storage if group_kind == "block" else None,
            )

        assert qualified_result[0] == common_result[0] == "sorted"
        qualified_call, common_call = observed[-2:]
        assert qualified_call["group"] == common_call["group"]
        assert qualified_call["keys"] is common_call["keys"] is keys
        assert qualified_call["descending"] is common_call["descending"] is descending
        assert (
            qualified_call["valid_items"] == common_call["valid_items"] == valid_items
        )
        assert (
            qualified_call["oob_default"] == common_call["oob_default"] == oob_default
        )
        assert qualified_call["temp_storage"] is common_call["temp_storage"]
        assert qualified_call["source"] == common_call["source"] == "cutlass_root"

        plan = _group_merge_sort._make_group_merge_sort_plan(
            group=qualified_call["group"],
            launch=launch,
            key_dtype=Int32,
            value_dtype=None,
            items_per_thread=3,
            descending=descending,
            valid_items=valid_items,
            oob_default=oob_default,
        ).require_supported()
        assert plan.target is (
            GroupLoweringTarget.CUB_BLOCK
            if group_kind == "block"
            else GroupLoweringTarget.CUB_WARP
        )
        assert plan.implementation.struct_name == (
            "BlockMergeSort" if group_kind == "block" else "WarpMergeSort"
        )
        assert plan.implementation.method_name == "Sort"
        assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 3

    assert len(observed) == 2 * len(routes)


def test_common_merge_sort_rejects_lossy_sentinel_before_cutlass_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32, Uint32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import root_api
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider

    provider_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_merge_sort",
        lambda **kwargs: provider_calls.append(kwargs),
    )
    keys = cutlass_coop.ThreadData.from_values(3, 1, dtype=Int32)
    unsigned_keys = cutlass_coop.ThreadData.from_values(3, 1, dtype=Uint32)

    with root_api._compiler_scope("cuda.coop.cutlass"):
        block = coop.this_block()
        with pytest.raises(
            TypeError,
            match=(
                r"cuda\.coop\.merge_sort_keys oob_default must have the same "
                r"integer dtype as keys \(int32\); got float"
            ),
        ):
            coop.merge_sort_keys(
                block,
                keys,
                valid_items=63,
                oob_default=1.5,
            )
        with pytest.raises(
            ValueError,
            match=(
                r"cuda\.coop\.merge_sort_keys oob_default=2147483648 is not "
                r"representable in keys dtype int32"
            ),
        ):
            coop.merge_sort_keys(
                block,
                keys,
                valid_items=63,
                oob_default=1 << 31,
            )
        for sentinel in (-1, 1 << 32):
            with pytest.raises(
                ValueError,
                match=(
                    rf"cuda\.coop\.merge_sort_keys oob_default={sentinel} is "
                    r"not representable in keys dtype uint32"
                ),
            ):
                coop.merge_sort_keys(
                    block,
                    unsigned_keys,
                    valid_items=63,
                    oob_default=sentinel,
                )

    assert provider_calls == []
    assert root_api._common_root_operation_name() is None


@pytest.mark.parametrize(
    ("ordinary_dtype", "compiler_dtype"),
    [
        (int, "Int32"),
        (np.uint32, "Uint32"),
        (np.int64, "Int64"),
        (np.uint64, "Uint64"),
    ],
)
def test_merge_sort_resolves_portable_ordinary_thread_data_dtypes(
    ordinary_dtype,
    compiler_dtype,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    typing = pytest.importorskip("cutlass.base_dsl.typing")

    import cuda.coop.cutlass as cutlass_coop
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider

    resolved_dtype = getattr(typing, compiler_dtype)
    keys = cutlass_coop.ThreadData.from_values(
        resolved_dtype(3),
        resolved_dtype(1),
        dtype=ordinary_dtype,
    )
    key_type, key_values, key_data, *_ = provider._resolve_inputs(
        group=cutlass_coop.this_block(),
        keys=keys,
        values=None,
    )

    assert key_type is resolved_dtype
    assert key_values == keys.values("merge_sort_keys")
    assert key_data is keys


@pytest.mark.evidence_for(
    "group.merge_sort_pairs", backend="cutlass", evidence="lowering"
)
def test_common_and_qualified_merge_sort_pairs_share_provider_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Float64, Int32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import LaunchFacts, root_api
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider
    from cuda.coop.cutlass._dsl import _launch

    launch = LaunchFacts(exact_block_dim=(32, 1, 1))
    monkeypatch.setattr(_launch, "infer_launch_facts", lambda *_args, **_kwargs: launch)

    keys = cutlass_coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    values = cutlass_coop.ThreadData.from_values(
        Float64(30.5), Float64(10.5), dtype=Float64
    )
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_merge_sort",
        lambda **kwargs: observed.append(kwargs) or (keys, values),
    )
    qualified = cutlass_coop.merge_sort_pairs(
        cutlass_coop.this_block(),
        keys,
        values,
        descending=True,
    )
    with root_api._compiler_scope("cuda.coop.cutlass"):
        common = coop.merge_sort_pairs(coop.this_block(), keys, values, descending=True)

    assert qualified == common == (keys, values)
    assert len(observed) == 2
    assert observed[0] == observed[1]
    assert observed[1]["keys"] is keys
    assert observed[1]["values"] is values

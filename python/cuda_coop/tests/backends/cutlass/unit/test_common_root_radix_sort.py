# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest


@pytest.mark.evidence_for(
    "group.radix_sort_keys", backend="cutlass", evidence="lowering"
)
def test_common_and_qualified_radix_sort_share_exact_block_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import GroupOperandKind, LaunchFacts, root_api
    from cuda.coop.cutlass import _group_radix
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider
    from cuda.coop.cutlass._dsl import _launch

    launch = LaunchFacts(exact_block_dim=(8, 4, 2))
    monkeypatch.setattr(_launch, "infer_launch_facts", lambda *_args, **_kwargs: launch)

    keys = cutlass_coop.ThreadData.from_values(3, -1, dtype=Int32)
    storage = cutlass_coop.TempStorage()
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_radix_sort_keys",
        lambda **kwargs: observed.append(kwargs) or ("sorted", len(observed)),
    )

    qualified_result = cutlass_coop.radix_sort_keys(
        cutlass_coop.this_block(),
        keys,
        begin_bit=8,
        descending=True,
        temp_storage=storage,
    )
    with root_api._compiler_scope("cuda.coop.cutlass"):
        common_result = coop.radix_sort_keys(
            coop.this_block(),
            keys,
            begin_bit=8,
            descending=True,
            temp_storage=storage,
        )

    assert qualified_result[0] == common_result[0] == "sorted"
    qualified_call, common_call = observed
    assert qualified_call["group"] == common_call["group"]
    assert qualified_call["keys"] is common_call["keys"] is keys
    assert qualified_call["begin_bit"] == common_call["begin_bit"] == 8
    assert qualified_call["end_bit"] is common_call["end_bit"] is None
    assert qualified_call["descending"] is common_call["descending"] is True
    assert qualified_call["temp_storage"] is common_call["temp_storage"] is storage
    assert qualified_call["source"] == common_call["source"] == "cutlass_root"

    plan = _group_radix._make_group_radix_sort_plan(
        group=common_call["group"],
        launch=launch,
        key_dtype=Int32,
        value_dtype=None,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        descending=True,
        key_bit_width=32,
        source="common_root_test",
    ).require_supported()
    assert plan.implementation.struct_name == "BlockRadixSort"
    assert plan.implementation.method_name == "SortDescending"
    assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 2


@pytest.mark.parametrize(
    ("ordinary_dtype", "compiler_dtype"),
    [
        (int, "Int32"),
        (np.int32, "Int32"),
        (np.uint32, "Uint32"),
        (np.int64, "Int64"),
        (np.uint64, "Uint64"),
    ],
)
def test_radix_sort_resolves_portable_ordinary_thread_data_dtypes(
    ordinary_dtype,
    compiler_dtype,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    typing = pytest.importorskip("cutlass.base_dsl.typing")

    import cuda.coop.cutlass as cutlass_coop
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider

    resolved_dtype = getattr(typing, compiler_dtype)
    keys = cutlass_coop.ThreadData.from_values(
        resolved_dtype(3),
        resolved_dtype(1),
        dtype=ordinary_dtype,
    )
    key_type, key_values, operand_kind, key_data = provider._operand(
        keys,
        allowed=provider._RADIX_KEY_TYPES,
        feature="radix_sort_keys",
    )

    assert key_type is resolved_dtype
    assert key_values == keys.values("radix_sort_keys")
    assert operand_kind.value == "array"
    assert key_data is keys


@pytest.mark.evidence_for(
    "group.radix_sort_pairs", backend="cutlass", evidence="lowering"
)
def test_common_and_qualified_radix_sort_pairs_share_provider_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32, Uint8

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import LaunchFacts, root_api
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider
    from cuda.coop.cutlass._dsl import _launch

    launch = LaunchFacts(exact_block_dim=(32, 1, 1))
    monkeypatch.setattr(_launch, "infer_launch_facts", lambda *_args, **_kwargs: launch)

    keys = cutlass_coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    values = cutlass_coop.ThreadData.from_values(Uint8(30), Uint8(10), dtype=Uint8)
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_radix_sort_pairs",
        lambda **kwargs: observed.append(kwargs) or (keys, values),
    )
    qualified = cutlass_coop.radix_sort_pairs(
        cutlass_coop.this_block(),
        keys,
        values,
        begin_bit=4,
        end_bit=20,
    )
    with root_api._compiler_scope("cuda.coop.cutlass"):
        common = coop.radix_sort_pairs(
            coop.this_block(), keys, values, begin_bit=4, end_bit=20
        )

    assert qualified == common == (keys, values)
    assert len(observed) == 2
    assert observed[0] == observed[1]
    assert observed[1]["keys"] is keys
    assert observed[1]["values"] is values

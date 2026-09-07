# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re

import numpy as np
import pytest


@pytest.mark.evidence_for("group.topk_max_keys", backend="cutlass", evidence="lowering")
@pytest.mark.evidence_for("group.topk_min_keys", backend="cutlass", evidence="lowering")
def test_common_and_qualified_topk_share_exact_block_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import LaunchFacts, root_api
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl.block import _provider as provider

    launch = LaunchFacts(exact_block_dim=(64, 1, 1))
    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: launch,
    )

    keys = cutlass_coop.ThreadData.from_values(
        Int32(3),
        Int32(1),
        dtype=int,
    )
    original = tuple(keys)
    observed: list[dict[str, object]] = []

    def capture(**kwargs):
        observed.append(kwargs)
        return cutlass_coop.ThreadData.from_values(
            Int32(9),
            Int32(7),
            dtype=int,
        )

    monkeypatch.setattr(provider, "provider_topk_keys", capture)

    for operation, descending in (
        ("topk_max_keys", True),
        ("topk_min_keys", False),
    ):
        qualified_result = getattr(cutlass_coop, operation)(
            cutlass_coop.this_block(),
            keys,
            7,
            valid_items=97,
            begin_bit=4,
            end_bit=20,
        )
        with root_api._compiler_scope("cuda.coop.cutlass"):
            common_result = getattr(coop, operation)(
                coop.this_block(),
                keys,
                7,
                valid_items=97,
                begin_bit=4,
                end_bit=20,
            )

        assert tuple(qualified_result) == tuple(common_result) == (9, 7)
        qualified_call, common_call = observed[-2:]
        assert qualified_call == common_call
        assert qualified_call == {
            "key": keys,
            "k": 7,
            "num_valid": 97,
            "begin_bit": 4,
            "end_bit": 20,
            "descending": descending,
            "block_threads": 64,
            "temp_storage_primitive_name": operation,
        }

    assert len(observed) == 4
    assert tuple(keys) == original


@pytest.mark.parametrize(
    ("ordinary_dtype", "compiler_dtype"),
    [
        (int, "Int32"),
        (float, "Float32"),
        (np.uint8, "Uint8"),
        (np.int32, "Int32"),
        (np.uint32, "Uint32"),
        (np.int64, "Int64"),
        (np.uint64, "Uint64"),
        (np.float32, "Float32"),
        (np.float64, "Float64"),
    ],
)
def test_topk_resolves_qualified_ordinary_thread_data_dtypes(
    ordinary_dtype,
    compiler_dtype,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    typing = pytest.importorskip("cutlass.base_dsl.typing")

    import cuda.coop.cutlass as cutlass_coop
    from cuda.coop.cutlass._dsl.block import _provider as provider

    resolved_dtype = getattr(typing, compiler_dtype)
    keys = cutlass_coop.ThreadData.from_values(
        resolved_dtype(3),
        resolved_dtype(1),
        dtype=ordinary_dtype,
    )
    key_type, key_values = provider._resolve_topk_thread_data_value_type(
        keys,
        allowed=provider._TOPK_KEY_TYPES,
        feature="topk_keys",
    )

    assert key_type is resolved_dtype
    assert key_values == keys.values("topk_keys")


@pytest.mark.parametrize(
    ("value", "compiler_dtype"),
    [
        (3, "Int32"),
        (3.5, "Float32"),
        (np.uint8(3), "Uint8"),
        (np.int32(3), "Int32"),
        (np.uint32(3), "Uint32"),
        (np.int64(3), "Int64"),
        (np.uint64(3), "Uint64"),
        (np.float32(3.5), "Float32"),
        (np.float64(3.5), "Float64"),
        (np.dtype(np.float64), "Float64"),
    ],
)
def test_topk_resolves_qualified_ordinary_scalar_dtypes(
    value,
    compiler_dtype,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    typing = pytest.importorskip("cutlass.base_dsl.typing")

    from cuda.coop.cutlass._dsl.block import _provider as provider

    assert provider._resolve_topk_type(
        value,
        allowed=provider._TOPK_KEY_TYPES,
        feature="topk_keys",
    ) is getattr(typing, compiler_dtype)


def test_generated_topk_provider_proves_planned_scratch_is_sufficient() -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._dsl.block import _provider as provider

    request = provider._ShimRequest(
        kind="topk_keys",
        op="max",
        key_type=Int32,
        items_per_thread=2,
        block_threads=64,
    )
    source = "\n".join(provider._render_block_bundle_request(request))
    expected_size, expected_alignment = provider._topk_temp_storage_requirement(
        block_threads=64,
        items_per_thread=2,
        key_type=Int32,
    )

    assert re.search(
        rf"planned_temp_bytes =\n\s+{expected_size}ull;",
        source,
    )
    assert re.search(
        rf"planned_temp_alignment =\n\s+{expected_alignment}ull;",
        source,
    )
    assert "static_assert(planned_temp_bytes >= required_temp_bytes" in source
    assert "alignof(TempStorageT)" in source
    assert "alignof(KeyT)" in source


def test_generated_topk_pair_provider_proves_value_cache_alignment() -> None:
    pytest.importorskip("cutlass.cute.ffi")
    typing = pytest.importorskip("cutlass.base_dsl.typing")

    from cuda.coop.cutlass._dsl.block import _provider as provider

    request = provider._ShimRequest(
        kind="topk_pair_keys",
        op="min",
        key_type=typing.Int32,
        pair_value_type=typing.Float64,
        items_per_thread=2,
        block_threads=64,
    )
    source = "\n".join(provider._render_block_bundle_request(request))

    assert "alignof(ValueT)" in source
    assert "CuTe TopK scratch alignment is weaker than the value cache" in source


@pytest.mark.evidence_for(
    "group.topk_max_pairs", backend="cutlass", evidence="lowering"
)
@pytest.mark.evidence_for(
    "group.topk_min_pairs", backend="cutlass", evidence="lowering"
)
def test_common_and_qualified_topk_pairs_share_provider_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cutlass.base_dsl.typing import Float64, Int32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import LaunchFacts, root_api
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl.block import _provider as provider

    launch = LaunchFacts(exact_block_dim=(32, 1, 1))
    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: launch,
    )

    keys = cutlass_coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    values = cutlass_coop.ThreadData.from_values(
        Float64(30.5), Float64(10.5), dtype=Float64
    )
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_topk_pairs",
        lambda **kwargs: observed.append(kwargs) or (keys, values),
    )
    for operation in ("topk_max_pairs", "topk_min_pairs"):
        qualified = getattr(cutlass_coop, operation)(
            cutlass_coop.this_block(),
            keys,
            values,
            1,
        )
        with root_api._compiler_scope("cuda.coop.cutlass"):
            common = getattr(coop, operation)(coop.this_block(), keys, values, 1)
        assert qualified == common == (keys, values)
        assert observed[-2] == observed[-1]
        assert observed[-1]["key"] is keys
        assert observed[-1]["value"] is values

    assert len(observed) == 4


def test_common_and_qualified_topk_pairs_reuse_explicit_temp_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cutlass.base_dsl.typing import Float64, Int32

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import LaunchFacts, root_api
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl import _single_phase
    from cuda.coop.cutlass._dsl.block import _provider as provider

    launch = LaunchFacts(exact_block_dim=(32, 1, 1))
    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: launch,
    )

    keys = cutlass_coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    values = cutlass_coop.ThreadData.from_values(
        Float64(30.5), Float64(10.5), dtype=Float64
    )
    storage = cutlass_coop.TempStorage(size_in_bytes=16_400, alignment=16)
    contexts = []

    def capture(**kwargs):
        contexts.append((_single_phase.get_active_single_phase_context(), kwargs))
        return keys, values

    monkeypatch.setattr(provider, "provider_topk_pairs", capture)

    qualified = cutlass_coop.topk_max_pairs(
        cutlass_coop.this_block(),
        keys,
        values,
        1,
        temp_storage=storage,
    )
    with root_api._compiler_scope("cuda.coop.cutlass"):
        common = coop.topk_max_pairs(
            coop.this_block(),
            keys,
            values,
            1,
            temp_storage=storage,
        )

    assert qualified == common == (keys, values)
    assert len(contexts) == 2
    for context, payload in contexts:
        assert context is not None
        assert context.temp_storage is storage
        assert "launch_metadata" not in payload

    assert [use.primitive_name for use in storage.uses] == [
        "topk_max_pairs",
        "topk_max_pairs",
    ]
    assert {use.byte_offset_in_bytes for use in storage.uses} == {0}
    assert storage.required_size_in_bytes <= storage.capacity_size_in_bytes


def test_thread_data_topk_keys_plan_exclusive_temp_storage_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl.block import _provider as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: LaunchFacts(exact_block_dim=(32, 1, 1)),
    )
    keys = coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    storage = coop.TempStorage(
        size_in_bytes=16_384,
        alignment=16,
        sharing="exclusive",
    )
    monkeypatch.setattr(provider, "provider_topk_keys", lambda **_kwargs: keys)

    coop.topk_max_keys(coop.this_block(), keys, 1, temp_storage=storage)
    coop.topk_min_keys(coop.this_block(), keys, 1, temp_storage=storage)

    expected_size, expected_alignment = provider._topk_temp_storage_requirement(
        block_threads=32,
        items_per_thread=2,
        key_type=Int32,
    )
    assert [
        (use.required_size_in_bytes, use.required_alignment) for use in storage.uses
    ] == [
        (expected_size, expected_alignment),
        (expected_size, expected_alignment),
    ]
    assert [use.byte_offset_in_bytes for use in storage.uses] == [0, expected_size]


@pytest.mark.parametrize(
    ("values", "compiler_dtype"),
    [
        ((3, 1), "Int32"),
        ((np.int64(3), np.int64(1)), "Int64"),
    ],
)
def test_ordinary_thread_data_topk_keys_use_explicit_temp_storage(
    monkeypatch: pytest.MonkeyPatch,
    values,
    compiler_dtype,
) -> None:
    typing = pytest.importorskip("cutlass.base_dsl.typing")

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl.block import _provider as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: LaunchFacts(exact_block_dim=(32, 1, 1)),
    )
    keys = coop.ThreadData.from_values(*values)
    storage = coop.TempStorage(size_in_bytes=16_400, alignment=16)
    monkeypatch.setattr(provider, "provider_topk_keys", lambda **_kwargs: keys)

    result = coop.topk_max_keys(
        coop.this_block(),
        keys,
        1,
        temp_storage=storage,
    )

    expected_size, expected_alignment = provider._topk_temp_storage_requirement(
        block_threads=32,
        items_per_thread=2,
        key_type=getattr(typing, compiler_dtype),
    )
    assert result is keys
    assert [
        (use.required_size_in_bytes, use.required_alignment) for use in storage.uses
    ] == [(expected_size, expected_alignment)]


def test_mixed_ordinary_thread_data_topk_pairs_use_explicit_temp_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cutlass.base_dsl.typing import Float64, Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl.block import _provider as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: LaunchFacts(exact_block_dim=(32, 1, 1)),
    )
    keys = coop.ThreadData.from_values(3, 1)
    values = coop.ThreadData.from_values(np.float64(3.5), np.float64(1.5))
    storage = coop.TempStorage(size_in_bytes=16_400, alignment=16)
    monkeypatch.setattr(
        provider,
        "provider_topk_pairs",
        lambda **_kwargs: (keys, values),
    )

    result = coop.topk_min_pairs(
        coop.this_block(),
        keys,
        values,
        1,
        temp_storage=storage,
    )

    expected_size, expected_alignment = provider._topk_temp_storage_requirement(
        block_threads=32,
        items_per_thread=2,
        key_type=Int32,
        value_type=Float64,
    )
    assert result == (keys, values)
    assert [
        (use.required_size_in_bytes, use.required_alignment) for use in storage.uses
    ] == [(expected_size, expected_alignment)]


def test_scalar_topk_pairs_use_explicit_temp_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cutlass.base_dsl.typing import Float64, Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl import _single_phase
    from cuda.coop.cutlass._dsl.block import _provider as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: LaunchFacts(exact_block_dim=(32, 1, 1)),
    )
    key = Int32(3)
    value = Float64(30.5)
    storage = coop.TempStorage(size_in_bytes=16_400, alignment=16)
    contexts = []

    def capture(**_kwargs):
        contexts.append(_single_phase.get_active_single_phase_context())
        return key, value

    monkeypatch.setattr(provider, "provider_topk_pairs", capture)

    result = coop.topk_min_pairs(
        coop.this_block(),
        key,
        value,
        1,
        temp_storage=storage,
    )

    assert result == (key, value)
    assert len(contexts) == 1
    assert contexts[0] is not None
    assert contexts[0].temp_storage is storage
    assert [use.primitive_name for use in storage.uses] == ["topk_min_pairs"]
    assert storage.required_size_in_bytes <= storage.capacity_size_in_bytes


@pytest.mark.parametrize(
    ("value", "compiler_dtype"),
    [
        (3, "Int32"),
        (3.5, "Float32"),
    ],
)
def test_ordinary_scalar_topk_pairs_use_explicit_temp_storage(
    monkeypatch: pytest.MonkeyPatch,
    value,
    compiler_dtype,
) -> None:
    typing = pytest.importorskip("cutlass.base_dsl.typing")

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl.block import _provider as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: LaunchFacts(exact_block_dim=(32, 1, 1)),
    )
    storage = coop.TempStorage(size_in_bytes=16_400, alignment=16)
    monkeypatch.setattr(
        provider,
        "provider_topk_pairs",
        lambda **_kwargs: (value, value),
    )

    result = coop.topk_max_pairs(
        coop.this_block(),
        value,
        value,
        1,
        temp_storage=storage,
    )

    expected_size, expected_alignment = provider._topk_temp_storage_requirement(
        block_threads=32,
        items_per_thread=1,
        key_type=getattr(typing, compiler_dtype),
        value_type=getattr(typing, compiler_dtype),
    )
    assert result == (value, value)
    assert [
        (use.required_size_in_bytes, use.required_alignment) for use in storage.uses
    ] == [(expected_size, expected_alignment)]


def test_topk_temp_storage_and_provider_session_roll_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._dsl.block import _provider as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: LaunchFacts(exact_block_dim=(32, 1, 1)),
    )
    storage = coop.TempStorage(size_in_bytes=16_400, alignment=16)
    provider_snapshot = object()
    restored = []
    monkeypatch.setattr(
        provider,
        "_snapshot_active_session_state",
        lambda: provider_snapshot,
    )
    monkeypatch.setattr(
        provider,
        "_restore_active_session_state",
        lambda snapshot: restored.append(snapshot),
    )

    def fail(**_kwargs):
        raise RuntimeError("provider failure")

    monkeypatch.setattr(provider, "provider_topk_keys", fail)

    with pytest.raises(RuntimeError, match="provider failure"):
        coop.topk_max_keys(
            coop.this_block(),
            Int32(3),
            1,
            temp_storage=storage,
        )

    assert storage.uses == ()
    assert restored == [provider_snapshot]


@pytest.mark.parametrize(
    ("storage_factory", "error_type", "message"),
    [
        (
            lambda coop: object(),
            TypeError,
            "expected temp_storage to be TempStorage or None",
        ),
        (
            lambda coop: coop.TempStorage(),
            NotImplementedError,
            "does not yet support inferred TempStorage",
        ),
    ],
)
def test_group_topk_rejects_invalid_cutlass_temp_storage(
    monkeypatch: pytest.MonkeyPatch,
    storage_factory,
    error_type,
    message,
) -> None:
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_topk

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: LaunchFacts(exact_block_dim=(32, 1, 1)),
    )

    with pytest.raises(error_type, match=message):
        coop.topk_max_keys(
            coop.this_block(),
            Int32(3),
            1,
            temp_storage=storage_factory(coop),
        )

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np
import pytest


def _launch_facts(block=(64, 1, 1)):
    from cuda.coop._core import LaunchFacts

    return LaunchFacts(exact_block_dim=block)


def test_common_and_qualified_topk_use_the_dedicated_provider(monkeypatch) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Float32, Int32

    import cuda.coop.cutlass as coop
    from cuda import coop as common_coop
    from cuda.coop._core.api import _dispatch as portable_dispatch
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._lowering import _topk as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: _launch_facts(),
    )
    keys = coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    values = coop.ThreadData.from_values(Float32(30.5), Float32(10.5), dtype=Float32)
    original_keys = tuple(keys)
    original_values = tuple(values)
    observed = []

    def capture_keys(**kwargs):
        observed.append(kwargs)
        return coop.ThreadData.from_values(Int32(9), Int32(7), dtype=Int32)

    def capture_pairs(**kwargs):
        observed.append(kwargs)
        return (
            coop.ThreadData.from_values(Int32(8), Int32(6), dtype=Int32),
            coop.ThreadData.from_values(Float32(80.5), Float32(60.5), dtype=Float32),
        )

    monkeypatch.setattr(provider, "provider_topk_keys", capture_keys)
    monkeypatch.setattr(provider, "provider_topk_pairs", capture_pairs)

    qualified = coop.topk_max_keys(
        coop.this_block(),
        keys,
        7,
        valid_items=97,
        begin_bit=4,
        end_bit=20,
    )
    with portable_dispatch._compiler_scope("cuda.coop.cutlass"):
        common = common_coop.topk_max_keys(
            common_coop.this_block(),
            keys,
            7,
            valid_items=97,
            begin_bit=4,
            end_bit=20,
        )
    assert tuple(qualified) == tuple(common) == (9, 7)
    assert qualified is not keys
    assert common is not keys
    assert observed[-2] == observed[-1]
    assert observed[-1]["descending"] is True

    qualified_pair = coop.topk_min_pairs(coop.this_block(), keys, values, 4)
    with portable_dispatch._compiler_scope("cuda.coop.cutlass"):
        common_pair = common_coop.topk_min_pairs(
            common_coop.this_block(), keys, values, 4
        )
    assert tuple(qualified_pair[0]) == tuple(common_pair[0]) == (8, 6)
    assert tuple(qualified_pair[1]) == tuple(common_pair[1]) == (80.5, 60.5)
    assert all(
        result is not source for result, source in zip(qualified_pair, (keys, values))
    )
    assert observed[-2] == observed[-1]
    assert observed[-1]["descending"] is False
    assert tuple(keys) == original_keys
    assert tuple(values) == original_values


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_topk_rejects_boolean_controls(value) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._lowering import _topk as provider

    for name in ("k", "num_valid", "begin_bit", "end_bit"):
        controls = {
            "k": 1,
            "num_valid": 64,
            "begin_bit": 0,
            "end_bit": 32,
        }
        controls[name] = value
        with pytest.raises(TypeError, match=rf"{name.replace('num_', '')}.*integer"):
            provider._validate_controls(
                **controls,
                key_type=Int32,
                tile_size=64,
            )


def test_topk_accepts_index_protocol_and_checks_static_ranges() -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._lowering import _topk as provider

    class Index:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    assert (
        provider._validate_controls(
            k=Index(4),
            num_valid=Index(63),
            begin_bit=Index(1),
            end_bit=None,
            key_type=Int32,
            tile_size=64,
        )
        == 32
    )
    with pytest.raises(ValueError, match="k must be <= valid_items"):
        provider._validate_controls(
            k=5,
            num_valid=4,
            begin_bit=0,
            end_bit=32,
            key_type=Int32,
            tile_size=64,
        )
    with pytest.raises(ValueError, match="end_bit must exceed begin_bit"):
        provider._validate_controls(
            k=1,
            num_valid=None,
            begin_bit=8,
            end_bit=8,
            key_type=Int32,
            tile_size=64,
        )


def test_topk_requires_a_complete_one_dimensional_block(monkeypatch) -> None:
    pytest.importorskip("cutlass.cute.ffi")

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass import _group_topk

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: _launch_facts((32, 2, 1)),
    )
    with pytest.raises(NotImplementedError, match="one-dimensional"):
        coop.topk_max_keys(coop.this_block(), 1, 1)

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: _launch_facts(),
    )
    with pytest.raises(NotImplementedError, match="only this_block"):
        coop.topk_max_keys(coop.this_warp(), 1, 1)


def test_topk_fixed_temp_storage_is_planned_transactionally(monkeypatch) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass import _group_topk
    from cuda.coop.cutlass._compiler._call_context import (
        get_active_single_phase_context,
    )
    from cuda.coop.cutlass._lowering import _topk as provider

    monkeypatch.setattr(
        _group_topk,
        "infer_launch_facts",
        lambda *_args, **_kwargs: _launch_facts(),
    )
    keys = coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    storage = coop.TempStorage(32_768, alignment=16)
    contexts = []

    def capture(**_kwargs):
        contexts.append(get_active_single_phase_context())
        return coop.ThreadData.from_values(Int32(9), Int32(7), dtype=Int32)

    monkeypatch.setattr(provider, "provider_topk_keys", capture)
    coop.topk_max_keys(coop.this_block(), keys, 1, temp_storage=storage)

    assert contexts[0] is not None
    assert contexts[0].temp_storage is storage
    assert [use.primitive_name for use in storage.uses] == ["topk_max_keys"]
    expected = provider._topk_temp_storage_requirement(
        block_threads=64,
        items_per_thread=2,
        key_type=Int32,
    )
    assert (
        storage.uses[0].required_size_in_bytes,
        storage.uses[0].required_alignment,
    ) == expected

    with pytest.raises(NotImplementedError, match="inferred TempStorage"):
        coop.topk_max_keys(
            coop.this_block(),
            keys,
            1,
            temp_storage=coop.TempStorage(),
        )


def test_dedicated_provider_source_has_one_cub_call_and_fresh_outputs() -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Float32, Int32

    from cuda.coop._core.block import ArgumentBinding, make_block_topk_spec
    from cuda.coop.cutlass._compiler import _rendering as provider_rendering
    from cuda.coop.cutlass._lowering import _topk as provider

    request = provider._CubTopKRequest(
        core_spec=make_block_topk_spec(
            key_dtype=Int32,
            value_dtype=Float32,
            block_dim=(64, 1, 1),
            items_per_thread=2,
            selection="min",
            num_valid=ArgumentBinding.runtime(),
            begin_bit=ArgumentBinding.runtime(),
            end_bit=ArgumentBinding.runtime(),
        ),
        key_type=Int32,
        value_type=Float32,
        external_scratch=True,
    )
    source = provider_rendering.render_bundle_source(
        [request],
        scope="cuda.coop.cutlass",
        render_local_request=lambda unexpected: pytest.fail(str(unexpected)),
    )

    assert source.count("#include <cub/block/block_topk.cuh>") == 1
    assert source.count("implementation_type(storage).min_pairs_partial(") == 1
    assert "result_keys[i] = keys[i]" in source
    assert "result_values[i] = values[i]" in source
    assert "#include <cub/block/block_radix" not in source
    assert "BlockRadix" not in source

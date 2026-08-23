# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    ArgumentKind,
    Array,
    Dependency,
    ParameterRole,
    TempStorageParameter,
)
from cuda.coop._core.warp import (
    WarpExchangeMode,
    WarpExchangeValueForm,
    make_warp_exchange_spec,
)


def test_warp_exchange_out_of_place_semantics():
    spec = make_warp_exchange_spec(
        dtype="i32",
        items_per_thread=4,
        threads_in_warp=16,
        mode="striped_to_blocked",
    )

    assert spec.mode is WarpExchangeMode.STRIPED_TO_BLOCKED
    assert spec.value_form is WarpExchangeValueForm.OUT_OF_PLACE
    assert spec.method_name == "StripedToBlocked"
    assert not spec.uses_ranks
    assert spec.specialization.template_arguments == {
        "T": "i32",
        "ITEMS_PER_THREAD": 4,
        "LOGICAL_WARP_THREADS": 16,
        "WARP_EXCHANGE_ALGORITHM": "::cub::WARP_EXCHANGE_SMEM",
    }
    assert spec.specialization.symbol_mangling_inputs[:2] == (
        "warp_exchange",
        "StripedToBlocked",
    )
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="input_items",
            ),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="output_items",
                is_output=True,
                is_return=False,
            ),
        ),
    )
    assert [
        (item.kind, item.role) for item in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]


def test_warp_exchange_scatter_can_expose_both_value_forms():
    spec = make_warp_exchange_spec(
        dtype="f32",
        items_per_thread=2,
        threads_in_warp=8,
        mode=WarpExchangeMode.SCATTER_TO_STRIPED,
        value_form=WarpExchangeValueForm.BOTH,
        rank_dtype="i32",
    )

    assert spec.uses_ranks
    assert spec.rank_dtype == "i32"
    assert spec.specialization.template_arguments["OffsetT"] == "i32"
    assert [
        [parameter.name for parameter in method]
        for method in spec.specialization.parameters
    ] == [
        ["temp_storage", "items", "ranks"],
        ["temp_storage", "input_items", "output_items", "ranks"],
    ]
    assert spec.specialization.parameters[0][1].is_inout
    assert spec.specialization.parameters[0][1].is_return is False
    assert spec.specialization.parameters[1][2].is_output
    assert spec.specialization.parameters[1][2].is_return is False


def test_warp_exchange_semantic_identity_tracks_mode_form_and_rank_dtype():
    def exchange(*, mode="scatter_to_striped", form="out_of_place", rank="i32"):
        return make_warp_exchange_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=16,
            mode=mode,
            value_form=form,
            rank_dtype=rank if mode == "scatter_to_striped" else None,
        )

    assert exchange().semantic_key == exchange().semantic_key
    assert exchange().semantic_key != exchange(form="in_place").semantic_key
    assert exchange().semantic_key != exchange(rank="i64").semantic_key
    assert exchange().semantic_key != exchange(mode="blocked_to_striped").semantic_key


def test_warp_exchange_rejects_invalid_form_and_rank_combinations():
    with pytest.raises(ValueError, match="rank_dtype is required"):
        make_warp_exchange_spec(
            dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            mode="scatter_to_striped",
        )
    with pytest.raises(ValueError, match="only valid for scatter"):
        make_warp_exchange_spec(
            dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            mode="blocked_to_striped",
            rank_dtype="i32",
        )
    with pytest.raises(ValueError, match="in-place overloads"):
        make_warp_exchange_spec(
            dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            mode="striped_to_blocked",
            value_form="in_place",
        )


@pytest.mark.parametrize(
    ("items_per_thread", "threads_in_warp"),
    [(0, 32), (1, 3), (True, 32)],
)
def test_warp_exchange_rejects_invalid_shapes(items_per_thread, threads_in_warp):
    with pytest.raises(ValueError):
        make_warp_exchange_spec(
            dtype="i32",
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            mode="striped_to_blocked",
        )

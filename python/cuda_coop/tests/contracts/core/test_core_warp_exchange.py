# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core import ArgumentKind, ParameterRole
from cuda.coop._core.warp import (
    WarpExchangeMode,
    WarpExchangeValueForm,
    make_warp_exchange_spec,
)


@pytest.mark.parametrize("threads_in_warp", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize(
    ("mode", "method"),
    [
        ("striped_to_blocked", "StripedToBlocked"),
        ("blocked_to_striped", "BlockedToStriped"),
    ],
)
def test_warp_exchange_supports_each_cub_logical_width(
    threads_in_warp,
    mode,
    method,
):
    spec = make_warp_exchange_spec(
        dtype="i32",
        items_per_thread=4,
        threads_in_warp=threads_in_warp,
        mode=mode,
    )

    assert spec.mode is WarpExchangeMode(mode)
    assert spec.value_form is WarpExchangeValueForm.OUT_OF_PLACE
    assert spec.method_name == method
    assert spec.specialization.template_arguments == {
        "T": "i32",
        "ITEMS_PER_THREAD": 4,
        "LOGICAL_WARP_THREADS": threads_in_warp,
        "WARP_EXCHANGE_ALGORITHM": "::cub::WARP_EXCHANGE_SMEM",
    }
    assert [entry.role for entry in spec.specialization.classify_method()] == [
        ParameterRole.TEMP_STORAGE,
        ParameterRole.INPUT,
        ParameterRole.OUTPUT,
    ]
    assert all(
        entry.kind is ArgumentKind.RUNTIME
        for entry in spec.specialization.classify_method()
    )


def test_warp_exchange_scatter_exposes_both_cub_forms():
    spec = make_warp_exchange_spec(
        dtype="f32",
        items_per_thread=2,
        threads_in_warp=8,
        mode="scatter_to_striped",
        value_form="both",
        rank_dtype="i32",
    )

    assert spec.uses_ranks
    assert spec.specialization.template_arguments["OffsetT"] == "i32"
    assert [
        [parameter.name for parameter in method]
        for method in spec.specialization.parameters
    ] == [
        ["temp_storage", "items", "ranks"],
        ["temp_storage", "input_items", "output_items", "ranks"],
    ]
    assert spec.specialization.parameters[0][1].is_inout
    assert spec.specialization.parameters[1][2].is_output


def test_warp_exchange_identity_tracks_width_mode_form_and_rank_dtype():
    def exchange(**overrides):
        options = {
            "dtype": "i32",
            "items_per_thread": 2,
            "threads_in_warp": 16,
            "mode": "scatter_to_striped",
            "value_form": "out_of_place",
            "rank_dtype": "i32",
        }
        options.update(overrides)
        return make_warp_exchange_spec(**options)

    assert exchange().semantic_key == exchange().semantic_key
    assert exchange().semantic_key != exchange(dtype="i64").semantic_key
    assert exchange().semantic_key != exchange(items_per_thread=3).semantic_key
    assert exchange().semantic_key != exchange(threads_in_warp=8).semantic_key
    assert exchange().semantic_key != exchange(value_form="in_place").semantic_key
    assert exchange().semantic_key != exchange(rank_dtype="i64").semantic_key


@pytest.mark.parametrize("threads_in_warp", [True, 0, -1, 3, 6, 24, 31, 33, 8.0])
def test_warp_exchange_rejects_invalid_widths(threads_in_warp):
    with pytest.raises(ValueError, match="threads_in_warp in"):
        make_warp_exchange_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=threads_in_warp,
            mode="striped_to_blocked",
        )


def test_warp_exchange_rejects_inconsistent_forms_and_ranks():
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

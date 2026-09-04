# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core import ArgumentKind, Array, Dependency, ParameterRole
from cuda.coop._core.block import (
    BlockExchangeMode,
    BlockExchangeValueForm,
    make_block_exchange_semantics,
    make_block_exchange_spec,
)


@pytest.mark.parametrize(
    ("mode", "method", "uses_ranks", "uses_flags"),
    [
        ("striped_to_blocked", "StripedToBlocked", False, False),
        ("blocked_to_striped", "BlockedToStriped", False, False),
        ("warp_striped_to_blocked", "WarpStripedToBlocked", False, False),
        ("blocked_to_warp_striped", "BlockedToWarpStriped", False, False),
        ("scatter_to_blocked", "ScatterToBlocked", True, False),
        ("scatter_to_striped", "ScatterToStriped", True, False),
        ("scatter_to_striped_guarded", "ScatterToStripedGuarded", True, False),
        ("scatter_to_striped_flagged", "ScatterToStripedFlagged", True, True),
    ],
)
def test_block_exchange_modes_own_their_cub_contract(
    mode,
    method,
    uses_ranks,
    uses_flags,
):
    semantics = make_block_exchange_semantics(
        dtype="i32",
        items_per_thread=2,
        mode=mode,
        rank_dtype="i32" if uses_ranks else None,
        valid_flag_dtype="u8" if uses_flags else None,
    )

    assert semantics.method_name == method
    assert BlockExchangeMode.from_cub_method_name(method) is semantics.mode
    assert semantics.uses_ranks is uses_ranks
    assert semantics.uses_valid_flags is uses_flags
    assert semantics.value_form is BlockExchangeValueForm.OUT_OF_PLACE


def test_block_exchange_flagged_forms_preserve_parameters_and_roles():
    spec = make_block_exchange_spec(
        dtype="i32",
        block_dim=(8, 4, 2),
        items_per_thread=3,
        mode="scatter_to_striped_flagged",
        value_form="both",
        rank_dtype="i16",
        valid_flag_dtype="u8",
    )

    assert spec.specialization.template_arguments == {
        "T": "i32",
        "BLOCK_DIM_X": 8,
        "ITEMS_PER_THREAD": 3,
        "WARP_TIME_SLICING": 0,
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
        "OffsetT": "i16",
        "ValidFlag": "u8",
    }
    assert [
        [parameter.name for parameter in method]
        for method in spec.specialization.parameters
    ] == [
        ["temp_storage", "input_items", "ranks", "valid_flags"],
        [
            "temp_storage",
            "input_items",
            "output_items",
            "ranks",
            "valid_flags",
        ],
    ]
    assert spec.specialization.parameters[0][1] == Array(
        Dependency("T"),
        Dependency("ITEMS_PER_THREAD"),
        name="input_items",
        is_inout=True,
        is_return=False,
    )
    assert [entry.role for entry in spec.specialization.classify_method(0)] == [
        ParameterRole.TEMP_STORAGE,
        ParameterRole.INOUT,
        ParameterRole.INPUT,
        ParameterRole.INPUT,
    ]
    assert [entry.role for entry in spec.specialization.classify_method(1)] == [
        ParameterRole.TEMP_STORAGE,
        ParameterRole.INPUT,
        ParameterRole.OUTPUT,
        ParameterRole.INPUT,
        ParameterRole.INPUT,
    ]
    assert all(
        entry.kind is ArgumentKind.RUNTIME
        for method_index in range(2)
        for entry in spec.specialization.classify_method(method_index)
    )


def test_block_exchange_identity_tracks_policy_and_specialization_shape():
    def make(**overrides):
        options = {
            "dtype": "i32",
            "items_per_thread": 2,
            "mode": "scatter_to_striped",
            "value_form": "out_of_place",
            "rank_dtype": "i32",
        }
        options.update(overrides)
        return make_block_exchange_semantics(**options)

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(dtype="i64").semantic_key
    assert make().semantic_key != make(items_per_thread=3).semantic_key
    assert make().semantic_key != make(value_form="in_place").semantic_key
    assert make().semantic_key != make(warp_time_slicing=True).semantic_key
    assert make().semantic_key != make(rank_dtype="i64").semantic_key

    spec = make_block_exchange_spec(
        dtype="i32",
        block_dim=(16, 2, 1),
        items_per_thread=2,
        mode="striped_to_blocked",
        warp_time_slicing=True,
    )
    assert spec.specialization.includes == ("cub/block/block_exchange.cuh",)
    assert spec.specialization.metadata["scope"] == "block"
    assert spec.specialization.template_arguments["WARP_TIME_SLICING"] == 1


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_block_exchange_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="positive integer"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=items_per_thread,
            mode="striped_to_blocked",
        )


def test_block_exchange_rejects_inconsistent_options():
    with pytest.raises(ValueError, match="dtype must be provided"):
        make_block_exchange_semantics(
            dtype=None,
            items_per_thread=1,
            mode="striped_to_blocked",
        )
    with pytest.raises(ValueError, match="boolean"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=1,
            mode="striped_to_blocked",
            warp_time_slicing=1,
        )
    with pytest.raises(ValueError, match="rank_dtype is required"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=1,
            mode="scatter_to_striped",
        )
    with pytest.raises(ValueError, match="rank_dtype is only valid"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=1,
            mode="striped_to_blocked",
            rank_dtype="i32",
        )
    with pytest.raises(ValueError, match="valid_flag_dtype is required"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=1,
            mode="scatter_to_striped_flagged",
            rank_dtype="i32",
        )
    with pytest.raises(ValueError, match="valid_flag_dtype is only valid"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=1,
            mode="scatter_to_striped",
            rank_dtype="i32",
            valid_flag_dtype="u8",
        )
    with pytest.raises(ValueError, match="guarded or flagged"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=1,
            mode="scatter_to_striped_guarded",
            warp_time_slicing=True,
            rank_dtype="i32",
        )
    with pytest.raises(ValueError, match="block_dim"):
        make_block_exchange_spec(
            dtype="i32",
            block_dim=(32, 0, 1),
            items_per_thread=1,
            mode="striped_to_blocked",
        )

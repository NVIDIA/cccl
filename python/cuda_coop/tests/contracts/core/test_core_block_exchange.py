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
from cuda.coop._core.block import (
    BlockExchangeMode,
    BlockExchangeValueForm,
    make_block_exchange_semantics,
    make_block_exchange_spec,
)


@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        ("striped_to_blocked", "StripedToBlocked"),
        ("blocked_to_striped", "BlockedToStriped"),
        ("warp_striped_to_blocked", "WarpStripedToBlocked"),
        ("blocked_to_warp_striped", "BlockedToWarpStriped"),
        ("scatter_to_blocked", "ScatterToBlocked"),
        ("scatter_to_striped", "ScatterToStriped"),
        ("scatter_to_striped_guarded", "ScatterToStripedGuarded"),
        ("scatter_to_striped_flagged", "ScatterToStripedFlagged"),
    ],
)
def test_block_exchange_mode_selects_cub_method(mode, method_name):
    is_scatter = mode.startswith("scatter_")
    is_flagged = mode == "scatter_to_striped_flagged"
    call = make_block_exchange_semantics(
        dtype="i32",
        items_per_thread=2,
        mode=mode,
        rank_dtype="i16" if is_scatter else None,
        valid_flag_dtype="u8" if is_flagged else None,
    )

    assert call.method_name == method_name
    assert call.mode.cub_method_name == method_name
    assert BlockExchangeMode.from_cub_method_name(method_name) is call.mode
    assert call.uses_ranks is is_scatter
    assert call.uses_valid_flags is is_flagged
    assert call.value_form is BlockExchangeValueForm.OUT_OF_PLACE


def test_block_exchange_flagged_both_forms_preserve_parameter_order_and_roles():
    call = make_block_exchange_semantics(
        dtype="i32",
        items_per_thread=3,
        mode="scatter_to_striped_flagged",
        value_form="both",
        rank_dtype="i16",
        valid_flag_dtype="u8",
    )

    assert call.mode is BlockExchangeMode.SCATTER_TO_STRIPED_FLAGGED
    assert call.value_form is BlockExchangeValueForm.BOTH
    assert not call.warp_time_slicing
    assert call.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="input_items",
                is_inout=True,
                is_return=False,
            ),
            Array(
                Dependency("OffsetT"),
                Dependency("ITEMS_PER_THREAD"),
                name="ranks",
            ),
            Array(
                Dependency("ValidFlag"),
                Dependency("ITEMS_PER_THREAD"),
                name="valid_flags",
            ),
        ),
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
            Array(
                Dependency("OffsetT"),
                Dependency("ITEMS_PER_THREAD"),
                name="ranks",
            ),
            Array(
                Dependency("ValidFlag"),
                Dependency("ITEMS_PER_THREAD"),
                name="valid_flags",
            ),
        ),
    )

    spec = make_block_exchange_spec(
        dtype="i32",
        block_dim=(16, 2, 1),
        items_per_thread=3,
        mode=call.mode,
        value_form=call.value_form,
        rank_dtype="i16",
        valid_flag_dtype="u8",
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method(0)
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INOUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
    ]
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method(1)
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
    ]


def test_block_exchange_spec_records_class_and_auxiliary_template_arguments():
    spec = make_block_exchange_spec(
        dtype="i32",
        block_dim=(8, 4, 2),
        items_per_thread=2,
        mode="scatter_to_blocked",
        value_form="in_place",
        rank_dtype="i64",
    )

    assert spec.block_dim == (8, 4, 2)
    assert spec.method_name == "ScatterToBlocked"
    assert spec.specialization.c_name == "block_exchange"
    assert spec.specialization.template_arguments == {
        "T": "i32",
        "BLOCK_DIM_X": 8,
        "ITEMS_PER_THREAD": 2,
        "WARP_TIME_SLICING": 0,
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
        "OffsetT": "i64",
    }
    assert spec.specialization.ordered_template_arguments == (
        ("T", "i32"),
        ("BLOCK_DIM_X", 8),
        ("ITEMS_PER_THREAD", 2),
        ("WARP_TIME_SLICING", 0),
        ("BLOCK_DIM_Y", 4),
        ("BLOCK_DIM_Z", 2),
    )
    assert spec.specialization.ordered_specialization_arguments[-1] == (
        "OffsetT",
        "i64",
    )


def test_block_exchange_semantic_identity_tracks_overload_and_policy():
    def make(**kwargs):
        options = {
            "dtype": "i32",
            "items_per_thread": 2,
            "mode": "scatter_to_striped",
            "value_form": "out_of_place",
            "rank_dtype": "i32",
        }
        options.update(kwargs)
        return make_block_exchange_semantics(**options)

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(value_form="in_place").semantic_key
    assert make().semantic_key != make(warp_time_slicing=True).semantic_key
    assert make().semantic_key != make(rank_dtype="i64").semantic_key
    assert make().semantic_key != make(items_per_thread=3).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_block_exchange_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
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
    with pytest.raises(ValueError, match="warp_time_slicing must be a boolean"):
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
    with pytest.raises(ValueError, match="block_dim"):
        make_block_exchange_spec(
            dtype="i32",
            block_dim=(32, 0, 1),
            items_per_thread=1,
            mode="striped_to_blocked",
        )


@pytest.mark.parametrize(
    ("mode", "valid_flag_dtype"),
    [
        ("scatter_to_striped_guarded", None),
        ("scatter_to_striped_flagged", "u8"),
    ],
)
def test_block_exchange_rejects_unsafe_time_sliced_scatter_modes(
    mode, valid_flag_dtype
):
    with pytest.raises(ValueError, match="guarded or flagged"):
        make_block_exchange_semantics(
            dtype="i32",
            items_per_thread=2,
            mode=mode,
            warp_time_slicing=True,
            rank_dtype="i32",
            valid_flag_dtype=valid_flag_dtype,
        )

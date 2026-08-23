# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import operator

import pytest

from cuda.coop._core import (
    INT32,
    ArgumentKind,
    Array,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
    Reference,
    TempStorageParameter,
)
from cuda.coop._core.block import (
    BlockAdjacentDifferenceBoundary,
    BlockDiscontinuityMode,
    BlockTileBoundary,
    make_block_discontinuity_semantics,
    make_block_discontinuity_spec,
)


def _python_flag(op=operator.ne):
    return PythonOperator(
        ret_dtype=Dependency("FlagT"),
        arg_dtypes=(Dependency("T"), Dependency("T")),
        op=op,
        name="flag_op",
    )


def test_shared_block_tile_boundary_supports_both_neighbor_directions():
    assert BlockAdjacentDifferenceBoundary is BlockTileBoundary
    assert (
        BlockTileBoundary.from_presence(predecessor=False, successor=False)
        is BlockTileBoundary.NONE
    )
    both = BlockTileBoundary.from_presence(predecessor=True, successor=True)
    assert both is BlockTileBoundary.BOTH
    assert both.has_predecessor
    assert both.has_successor


def test_heads_semantics_own_cub_method_and_parameter_roles():
    flag_op = _python_flag()
    spec = make_block_discontinuity_spec(
        dtype="i32",
        flag_dtype="u8",
        block_dim=(16, 2, 1),
        items_per_thread=3,
        mode="heads",
        flag_operator=flag_op,
    )

    assert spec.mode is BlockDiscontinuityMode.HEADS
    assert spec.boundary is BlockTileBoundary.NONE
    assert spec.method_name == "FlagHeads"
    assert spec.has_heads
    assert not spec.has_tails
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("FlagT"),
                Dependency("ITEMS_PER_THREAD"),
                name="head_flags",
                is_output=True,
                is_return=False,
            ),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="input_items",
            ),
            flag_op,
        ),
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.STATIC, ParameterRole.OPERATOR),
    ]
    assert spec.specialization.template_arguments == {
        "T": "i32",
        "BLOCK_DIM_X": 16,
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
        "FlagT": "u8",
        "ITEMS_PER_THREAD": 3,
    }


def test_heads_with_predecessor_preserves_cub_parameter_order():
    call = make_block_discontinuity_semantics(
        dtype="i32",
        flag_dtype="i32",
        items_per_thread=2,
        mode=BlockDiscontinuityMode.HEADS,
        flag_operator=_python_flag(),
        tile_predecessor_item=4,
    )

    assert call.boundary is BlockTileBoundary.PREDECESSOR
    assert call.has_tile_predecessor
    assert [parameter.name for parameter in call.parameters] == [
        "temp_storage",
        "head_flags",
        "input_items",
        "flag_op",
        "tile_predecessor_item",
    ]
    assert call.parameters[-1] == Reference(
        Dependency("T"), name="tile_predecessor_item"
    )


def test_tails_with_successor_preserves_cub_parameter_order():
    call = make_block_discontinuity_semantics(
        dtype="i32",
        flag_dtype="i32",
        items_per_thread=2,
        mode=BlockDiscontinuityMode.TAILS,
        flag_operator=CxxOperator(
            "::cuda::std::not_equal_to<T>",
            Dependency("T"),
            name="flag_op",
        ),
        tile_successor_item=9,
    )

    assert call.boundary is BlockTileBoundary.SUCCESSOR
    assert call.has_tile_successor
    assert [parameter.name for parameter in call.parameters] == [
        "temp_storage",
        "tail_flags",
        "input_items",
        "flag_op",
        "tile_successor_item",
    ]
    assert call.parameters[-1] == Reference(Dependency("T"), name="tile_successor_item")


@pytest.mark.parametrize(
    ("predecessor", "successor", "boundary", "names"),
    [
        (
            None,
            None,
            BlockTileBoundary.NONE,
            ["temp_storage", "head_flags", "tail_flags", "input_items", "flag_op"],
        ),
        (
            0,
            None,
            BlockTileBoundary.PREDECESSOR,
            [
                "temp_storage",
                "head_flags",
                "tile_predecessor_item",
                "tail_flags",
                "input_items",
                "flag_op",
            ],
        ),
        (
            None,
            9,
            BlockTileBoundary.SUCCESSOR,
            [
                "temp_storage",
                "head_flags",
                "tail_flags",
                "tile_successor_item",
                "input_items",
                "flag_op",
            ],
        ),
        (
            0,
            9,
            BlockTileBoundary.BOTH,
            [
                "temp_storage",
                "head_flags",
                "tile_predecessor_item",
                "tail_flags",
                "tile_successor_item",
                "input_items",
                "flag_op",
            ],
        ),
    ],
)
def test_heads_and_tails_boundary_matrix_matches_cub_overloads(
    predecessor, successor, boundary, names
):
    call = make_block_discontinuity_semantics(
        dtype="i32",
        flag_dtype="u8",
        items_per_thread=4,
        mode="heads_and_tails",
        flag_operator=_python_flag(),
        tile_predecessor_item=predecessor,
        tile_successor_item=successor,
    )

    assert call.method_name == "FlagHeadsAndTails"
    assert call.boundary is boundary
    assert call.has_heads
    assert call.has_tails
    assert [parameter.name for parameter in call.parameters] == names


def test_semantic_identity_tracks_mode_boundary_shape_flag_dtype_and_operator():
    def make(
        *,
        mode="heads",
        items=2,
        flag_dtype="u8",
        predecessor=False,
        op=operator.ne,
    ):
        return make_block_discontinuity_semantics(
            dtype="i32",
            flag_dtype=flag_dtype,
            items_per_thread=items,
            mode=mode,
            flag_operator=_python_flag(op),
            tile_predecessor_item=object() if predecessor else None,
        )

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(mode="heads_and_tails").semantic_key
    assert make().semantic_key != make(items=3).semantic_key
    assert make().semantic_key != make(flag_dtype="i32").semantic_key
    assert make().semantic_key != make(predecessor=True).semantic_key
    assert make().semantic_key != make(op=operator.eq).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        make_block_discontinuity_semantics(
            dtype="i32",
            flag_dtype=INT32,
            items_per_thread=items_per_thread,
            mode="heads",
            flag_operator=_python_flag(),
        )


def test_rejects_missing_types_operator_invalid_boundaries_and_block_shape():
    options = {
        "dtype": "i32",
        "flag_dtype": "u8",
        "items_per_thread": 1,
        "mode": "heads",
        "flag_operator": _python_flag(),
    }
    with pytest.raises(ValueError, match="dtype must be provided"):
        make_block_discontinuity_semantics(**{**options, "dtype": None})
    with pytest.raises(ValueError, match="flag dtype must be provided"):
        make_block_discontinuity_semantics(**{**options, "flag_dtype": None})
    with pytest.raises(TypeError, match="requires a flag operator"):
        make_block_discontinuity_semantics(**{**options, "flag_operator": None})
    with pytest.raises(ValueError, match="flag_op must be provided"):
        make_block_discontinuity_semantics(
            **{**options, "flag_operator": _python_flag(None)}
        )
    with pytest.raises(ValueError, match="tile_successor_item.*HEADS"):
        make_block_discontinuity_semantics(
            **options,
            tile_successor_item=0,
        )
    with pytest.raises(ValueError, match="tile_predecessor_item.*TAILS"):
        make_block_discontinuity_semantics(
            **{**options, "mode": "tails"},
            tile_predecessor_item=0,
        )
    with pytest.raises(ValueError, match="block_dim"):
        make_block_discontinuity_spec(
            dtype="i32",
            flag_dtype="u8",
            block_dim=(32, 0, 1),
            items_per_thread=1,
            mode="heads",
            flag_operator=_python_flag(),
        )

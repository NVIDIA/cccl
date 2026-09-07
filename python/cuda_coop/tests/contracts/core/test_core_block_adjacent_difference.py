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
    Value,
)
from cuda.coop._core.block import (
    BlockAdjacentDifferenceBoundary,
    BlockAdjacentDifferenceDirection,
    BlockAdjacentDifferenceTilePolicy,
    make_block_adjacent_difference_semantics,
    make_block_adjacent_difference_spec,
)


def _python_difference(op=operator.sub):
    return PythonOperator(
        ret_dtype=Dependency("T"),
        arg_dtypes=(Dependency("T"), Dependency("T")),
        op=op,
        name="difference_op",
    )


def test_left_full_tile_owns_cub_method_and_parameter_roles():
    difference = _python_difference()
    spec = make_block_adjacent_difference_spec(
        dtype="i32",
        block_dim=(16, 2, 1),
        items_per_thread=3,
        direction="left",
        difference_operator=difference,
    )

    assert spec.direction is BlockAdjacentDifferenceDirection.LEFT
    assert spec.tile_policy is BlockAdjacentDifferenceTilePolicy.FULL
    assert spec.boundary is BlockAdjacentDifferenceBoundary.NONE
    assert spec.method_name == "SubtractLeft"
    assert not spec.has_partial_tile
    assert not spec.has_boundary_item
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
            difference,
        ),
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.STATIC, ParameterRole.OPERATOR),
    ]


def test_left_partial_tile_with_predecessor_preserves_cub_parameter_order():
    spec = make_block_adjacent_difference_spec(
        dtype="i32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        direction=BlockAdjacentDifferenceDirection.LEFT,
        difference_operator=_python_difference(),
        valid_items=17,
        tile_predecessor_item=-1,
    )

    assert spec.method_name == "SubtractLeftPartialTile"
    assert spec.has_partial_tile
    assert spec.boundary is BlockAdjacentDifferenceBoundary.PREDECESSOR
    assert [parameter.name for parameter in spec.specialization.parameters[0]] == [
        "temp_storage",
        "input_items",
        "output_items",
        "difference_op",
        "valid_items",
        "tile_predecessor_item",
    ]
    assert spec.specialization.parameters[0][-2] == Value(INT32, name="valid_items")
    assert spec.specialization.parameters[0][-1] == Reference(
        Dependency("T"), name="tile_predecessor_item"
    )


def test_right_full_tile_accepts_successor_but_partial_tile_does_not():
    full = make_block_adjacent_difference_semantics(
        dtype="i64",
        items_per_thread=4,
        direction="right",
        difference_operator=CxxOperator(
            "::cuda::std::minus<T>",
            Dependency("T"),
            name="difference_op",
        ),
        tile_successor_item=0,
    )

    assert full.method_name == "SubtractRight"
    assert full.boundary is BlockAdjacentDifferenceBoundary.SUCCESSOR
    assert full.parameters[-1] == Reference(Dependency("T"), name="tile_successor_item")

    with pytest.raises(
        ValueError,
        match="tile_successor_item is not valid for SubtractRightPartialTile",
    ):
        make_block_adjacent_difference_semantics(
            dtype="i64",
            items_per_thread=4,
            direction="right",
            difference_operator=_python_difference(),
            valid_items=31,
            tile_successor_item=0,
        )


def test_semantic_identity_tracks_direction_tile_boundary_shape_and_operator():
    def make(
        *,
        direction="left",
        items=2,
        partial=False,
        predecessor=False,
        op=operator.sub,
    ):
        return make_block_adjacent_difference_semantics(
            dtype="i32",
            items_per_thread=items,
            direction=direction,
            difference_operator=_python_difference(op),
            valid_items=object() if partial else None,
            tile_predecessor_item=object() if predecessor else None,
        )

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(direction="right").semantic_key
    assert make().semantic_key != make(items=3).semantic_key
    assert make().semantic_key != make(partial=True).semantic_key
    assert make().semantic_key != make(predecessor=True).semantic_key
    assert make().semantic_key != make(op=operator.add).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        make_block_adjacent_difference_semantics(
            dtype="i32",
            items_per_thread=items_per_thread,
            direction="left",
            difference_operator=_python_difference(),
        )


def test_rejects_missing_dtype_operator_invalid_boundaries_and_block_shape():
    with pytest.raises(ValueError, match="dtype must be provided"):
        make_block_adjacent_difference_semantics(
            dtype=None,
            items_per_thread=1,
            direction="left",
            difference_operator=_python_difference(),
        )
    with pytest.raises(TypeError, match="requires a difference operator"):
        make_block_adjacent_difference_semantics(
            dtype="i32",
            items_per_thread=1,
            direction="left",
            difference_operator=None,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_block_adjacent_difference_semantics(
            dtype="i32",
            items_per_thread=1,
            direction="left",
            difference_operator=_python_difference(),
            tile_predecessor_item=0,
            tile_successor_item=0,
        )
    with pytest.raises(ValueError, match="tile_successor_item.*SubtractLeft"):
        make_block_adjacent_difference_semantics(
            dtype="i32",
            items_per_thread=1,
            direction="left",
            difference_operator=_python_difference(),
            tile_successor_item=0,
        )
    with pytest.raises(ValueError, match="tile_predecessor_item.*SubtractRight"):
        make_block_adjacent_difference_semantics(
            dtype="i32",
            items_per_thread=1,
            direction="right",
            difference_operator=_python_difference(),
            tile_predecessor_item=0,
        )
    with pytest.raises(ValueError, match="block_dim"):
        make_block_adjacent_difference_spec(
            dtype="i32",
            block_dim=(32, 0, 1),
            items_per_thread=1,
            direction="left",
            difference_operator=_python_difference(),
        )

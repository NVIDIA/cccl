# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import operator

import pytest

from cuda.coop._core import (
    INT8,
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
    BlockMergeSortPayload,
    BlockMergeSortTilePolicy,
    make_block_merge_sort_semantics,
    make_block_merge_sort_spec,
)


def _python_compare(op=operator.lt):
    return PythonOperator(
        ret_dtype=INT8,
        arg_dtypes=(Dependency("KeyT"), Dependency("KeyT")),
        op=op,
        name="compare_op",
    )


def test_block_merge_sort_keys_full_tile_semantics():
    compare = _python_compare()
    spec = make_block_merge_sort_spec(
        key_dtype="i32",
        block_dim=(16, 2, 1),
        items_per_thread=3,
        compare_operator=compare,
    )

    assert spec.payload is BlockMergeSortPayload.KEYS
    assert spec.tile_policy is BlockMergeSortTilePolicy.FULL
    assert not spec.has_values
    assert not spec.has_partial_tile
    assert spec.method_name == "Sort"
    assert spec.block_dim == (16, 2, 1)
    assert spec.items_per_thread == 3
    assert spec.compare_operator is compare
    assert spec.specialization.template_arguments == {
        "KeyT": "i32",
        "BLOCK_DIM_X": 16,
        "ITEMS_PER_THREAD": 3,
        "ValueT": "::cub::NullType",
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("KeyT"),
                Dependency("ITEMS_PER_THREAD"),
                name="keys",
                is_inout=True,
                is_return=False,
            ),
            compare,
        ),
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INOUT),
        (ArgumentKind.STATIC, ParameterRole.OPERATOR),
    ]


def test_block_merge_sort_pairs_partial_tile_semantics():
    spec = make_block_merge_sort_spec(
        key_dtype="i64",
        value_dtype="f32",
        block_dim=(8, 4, 1),
        items_per_thread=2,
        compare_operator=_python_compare(),
        valid_items=37,
        oob_default=99,
    )

    assert spec.payload is BlockMergeSortPayload.PAIRS
    assert spec.tile_policy is BlockMergeSortTilePolicy.PARTIAL
    assert spec.has_values
    assert spec.has_partial_tile
    assert spec.value_dtype == "f32"
    assert spec.specialization.template_arguments["ValueT"] == "f32"
    assert [parameter.name for parameter in spec.specialization.parameters[0]] == [
        "temp_storage",
        "keys",
        "values",
        "compare_op",
        "valid_items",
        "oob_default",
    ]
    assert spec.specialization.parameters[0][1].is_inout
    assert spec.specialization.parameters[0][2].is_inout
    assert spec.specialization.parameters[0][-2] == Value(INT32, name="valid_items")
    assert spec.specialization.parameters[0][-1] == Reference(
        Dependency("KeyT"), name="oob_default"
    )


def test_block_merge_sort_call_semantics_are_dimension_independent():
    call = make_block_merge_sort_semantics(
        key_dtype="i32",
        value_dtype="u64",
        items_per_thread=4,
        compare_operator=CxxOperator(
            "::cuda::std::less<KeyT>",
            Dependency("KeyT"),
            name="compare_op",
        ),
    )

    assert call.has_values
    assert not call.has_partial_tile
    assert call.items_per_thread == 4
    assert [parameter.name for parameter in call.parameters] == [
        "temp_storage",
        "keys",
        "values",
        "compare_op",
    ]


def test_block_merge_sort_semantic_identity_tracks_shape_payload_tile_and_operator():
    def make(*, items=2, value=None, partial=False, op=operator.lt):
        return make_block_merge_sort_spec(
            key_dtype="i32",
            value_dtype=value,
            block_dim=(32, 1, 1),
            items_per_thread=items,
            compare_operator=_python_compare(op),
            valid_items=object() if partial else None,
            oob_default=object() if partial else None,
        )

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(items=3).semantic_key
    assert make().semantic_key != make(value="f32").semantic_key
    assert make(partial=True).semantic_key == make(partial=True).semantic_key
    assert make().semantic_key != make(partial=True).semantic_key
    assert make().semantic_key != make(op=operator.gt).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_block_merge_sort_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        make_block_merge_sort_semantics(
            key_dtype="i32",
            items_per_thread=items_per_thread,
            compare_operator=_python_compare(),
        )


@pytest.mark.parametrize(
    ("valid_items", "oob_default"),
    [(7, None), (None, 99)],
)
def test_block_merge_sort_requires_complete_partial_tile_arguments(
    valid_items, oob_default
):
    with pytest.raises(
        ValueError, match="valid_items and oob_default must be provided together"
    ):
        make_block_merge_sort_semantics(
            key_dtype="i32",
            items_per_thread=1,
            compare_operator=_python_compare(),
            valid_items=valid_items,
            oob_default=oob_default,
        )


def test_block_merge_sort_rejects_missing_dtype_operator_and_block_shape():
    with pytest.raises(ValueError, match="key dtype must be provided"):
        make_block_merge_sort_semantics(
            key_dtype=None,
            items_per_thread=1,
            compare_operator=_python_compare(),
        )
    with pytest.raises(TypeError, match="requires a comparison operator"):
        make_block_merge_sort_semantics(
            key_dtype="i32",
            items_per_thread=1,
            compare_operator=None,
        )
    with pytest.raises(ValueError, match="compare_op must be provided"):
        make_block_merge_sort_semantics(
            key_dtype="i32",
            items_per_thread=1,
            compare_operator=_python_compare(None),
        )
    with pytest.raises(ValueError, match="block_dim"):
        make_block_merge_sort_spec(
            key_dtype="i32",
            block_dim=(32, 0, 1),
            items_per_thread=1,
            compare_operator=_python_compare(),
        )
    with pytest.raises(ValueError, match="power-of-two block thread count"):
        make_block_merge_sort_spec(
            key_dtype="i32",
            block_dim=(48, 1, 1),
            items_per_thread=1,
            compare_operator=_python_compare(),
        )

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    INT32,
    INT64,
    ArgumentKind,
    Array,
    Dependency,
    ParameterRole,
    Pointer,
    PointerOffset,
    Reference,
    TempStorageParameter,
    Value,
)
from cuda.coop._core.block import (
    BlockLoadStoreAlgorithm,
    BlockLoadStoreKind,
    make_block_load_spec,
    make_block_load_store_semantics,
    make_block_store_spec,
)


def test_block_load_full_tile_semantics():
    spec = make_block_load_spec(
        dtype="i32",
        block_dim=(16, 2, 1),
        items_per_thread=3,
        algorithm="striped",
    )

    assert spec.kind is BlockLoadStoreKind.LOAD
    assert spec.algorithm is BlockLoadStoreAlgorithm.STRIPED
    assert spec.algorithm_cpp == "::cub::BLOCK_LOAD_STRIPED"
    assert spec.block_dim == (16, 2, 1)
    assert spec.items_per_thread == 3
    assert spec.has_full_tile
    assert not spec.has_valid_items
    assert spec.specialization.template_arguments == {
        "T": "i32",
        "BLOCK_DIM_X": 16,
        "ITEMS_PER_THREAD": 3,
        "ALGORITHM": "::cub::BLOCK_LOAD_STRIPED",
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Pointer(
                Dependency("T"),
                name="src",
                is_array_pointer=True,
                restrict=True,
            ),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="dst",
                is_output=True,
                is_return=False,
            ),
        ),
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]


def test_block_load_partial_default_and_pointer_offset_overloads():
    spec = make_block_load_spec(
        dtype="f32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        algorithm="::cub::BLOCK_LOAD_TRANSPOSE",
        valid_items=True,
        oob_default=True,
        include_full_tile=True,
        include_pointer_offset=True,
    )

    assert spec.algorithm is BlockLoadStoreAlgorithm.TRANSPOSE
    assert spec.has_valid_items
    assert spec.has_oob_default
    assert spec.has_full_tile
    assert spec.has_pointer_offset
    assert [
        [parameter.name for parameter in method]
        for method in spec.specialization.parameters
    ] == [
        ["temp_storage", "src", "dst"],
        ["temp_storage", "src", "dst", "num_valid_items", "oob_default"],
        [
            "temp_storage",
            "src",
            "dst",
            "num_valid_items",
            "oob_default",
            "offset",
        ],
        ["temp_storage", "src", "dst", "offset"],
    ]
    assert spec.specialization.parameters[1][-2:] == (
        Value(INT32, name="num_valid_items"),
        Reference(Dependency("T"), name="oob_default"),
    )
    assert spec.specialization.parameters[2][-1] == PointerOffset(
        INT64,
        name="offset",
        pointer_arg_index=0,
    )


def test_block_store_partial_tile_semantics():
    spec = make_block_store_spec(
        dtype="i64",
        block_dim=(32, 1, 1),
        items_per_thread=4,
        algorithm="vectorize",
        valid_items=True,
    )

    assert spec.kind is BlockLoadStoreKind.STORE
    assert spec.algorithm is BlockLoadStoreAlgorithm.VECTORIZE
    assert spec.algorithm_cpp == "::cub::BLOCK_STORE_VECTORIZE"
    assert not spec.has_full_tile
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Pointer(
                Dependency("T"),
                name="dst",
                is_output=True,
                is_return=False,
                is_array_pointer=True,
                restrict=True,
            ),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="src",
            ),
            Value(INT32, name="num_valid_items"),
        ),
    )


def test_pointer_offsets_extend_only_selected_tile_signatures():
    spec = make_block_store_spec(
        dtype="i32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        algorithm="direct",
        valid_items=True,
        include_pointer_offset=True,
    )

    assert not spec.has_full_tile
    assert spec.has_pointer_offset
    assert [
        [parameter.name for parameter in method]
        for method in spec.specialization.parameters
    ] == [
        ["temp_storage", "dst", "src", "num_valid_items"],
        ["temp_storage", "dst", "src", "num_valid_items", "offset"],
    ]


def test_block_load_store_call_semantics_are_dimension_independent():
    call = make_block_load_store_semantics(
        kind="load",
        dtype="i32",
        items_per_thread=2,
        algorithm="direct",
        valid_items=True,
        oob_default=True,
    )

    assert call.method_name == "Load"
    assert call.algorithm_cpp == "::cub::BLOCK_LOAD_DIRECT"
    assert [parameter.name for parameter in call.parameters[0]] == [
        "temp_storage",
        "src",
        "dst",
        "num_valid_items",
        "oob_default",
    ]


def test_block_load_store_semantic_identity_tracks_kind_algorithm_and_shape():
    def make(*, kind="load", algorithm="direct", items=2, valid=False):
        return make_block_load_store_semantics(
            kind=kind,
            dtype="i32",
            items_per_thread=items,
            algorithm=algorithm,
            valid_items=valid,
            oob_default=valid and kind == "load",
        )

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(kind="store").semantic_key
    assert make().semantic_key != make(algorithm="striped").semantic_key
    assert make().semantic_key != make(items=3).semantic_key
    assert make().semantic_key != make(valid=True).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_block_load_store_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        make_block_load_store_semantics(
            kind="load",
            dtype="i32",
            items_per_thread=items_per_thread,
            algorithm="direct",
        )


def test_block_load_store_rejects_invalid_options():
    with pytest.raises(ValueError, match="unsupported BlockLoad algorithm"):
        make_block_load_store_semantics(
            kind="load",
            dtype="i32",
            items_per_thread=1,
            algorithm="unknown",
        )
    with pytest.raises(ValueError, match="only valid for BlockLoad"):
        make_block_load_store_semantics(
            kind="store",
            dtype="i32",
            items_per_thread=1,
            algorithm="direct",
            valid_items=True,
            oob_default=True,
        )
    with pytest.raises(ValueError, match="requires a valid_items"):
        make_block_load_store_semantics(
            kind="load",
            dtype="i32",
            items_per_thread=1,
            algorithm="direct",
            oob_default=True,
        )
    with pytest.raises(ValueError, match="include_full_tile"):
        make_block_load_store_semantics(
            kind="load",
            dtype="i32",
            items_per_thread=1,
            algorithm="direct",
            include_full_tile=True,
        )
    with pytest.raises(ValueError, match="block_dim"):
        make_block_store_spec(
            dtype="i32",
            block_dim=(32, 0, 1),
            items_per_thread=1,
            algorithm="direct",
        )

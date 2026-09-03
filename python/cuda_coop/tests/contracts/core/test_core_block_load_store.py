# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import (
    INT32,
    INT64,
    ArgumentBinding,
    ArgumentKind,
    Array,
    Dependency,
    ParameterRole,
    Pointer,
    PointerOffset,
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
        Value("f32", name="oob_default"),
    )
    assert spec.specialization.parameters[2][-1] == PointerOffset(
        INT64,
        name="offset",
        pointer_arg_index=0,
    )


def test_block_load_preserves_static_tile_controls_in_the_implementation_abi():
    spec = make_block_load_spec(
        dtype="f32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        algorithm="direct",
        valid_items=ArgumentBinding.static(17),
        oob_default=ArgumentBinding.static(0),
        include_pointer_offset=ArgumentBinding.static(4),
    )

    assert len(spec.specialization.parameters) == 1
    method = spec.specialization.parameters[0]
    controls = method[-3:]
    assert [parameter.name for parameter in controls] == [
        "num_valid_items",
        "oob_default",
        "offset",
    ]
    assert all(parameter.argument_kind is ArgumentKind.STATIC for parameter in controls)
    assert controls[-1].static_value == 4


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


def test_block_pointer_offset_identity_normalizes_numpy_integer():
    kwargs = {
        "kind": "load",
        "dtype": "i32",
        "items_per_thread": 2,
        "algorithm": "direct",
    }
    plain = make_block_load_store_semantics(
        **kwargs,
        include_pointer_offset=ArgumentBinding.static(4),
    )
    numpy = make_block_load_store_semantics(
        **kwargs,
        include_pointer_offset=ArgumentBinding.static(np.int64(4)),
    )

    assert numpy.pointer_offset == ArgumentBinding.static(4)
    assert plain.semantic_key == numpy.semantic_key


def test_block_provider_spec_rejects_negative_static_pointer_offset():
    with pytest.raises(ValueError, match="static pointer offset must be nonnegative"):
        make_block_store_spec(
            dtype=INT32,
            block_dim=(32, 1, 1),
            items_per_thread=1,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.static(-1),
        )


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


@pytest.mark.parametrize("block_dim", [(True, 1, 1), (1.5, 1, 1), ("2", 1, 1)])
def test_block_load_store_rejects_non_integral_block_dimensions(block_dim):
    with pytest.raises(ValueError, match="three positive dimensions"):
        make_block_load_spec(
            dtype="i32",
            block_dim=block_dim,
            items_per_thread=1,
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


@pytest.mark.parametrize("make_spec", [make_block_load_spec, make_block_store_spec])
@pytest.mark.parametrize("valid_items", [-1, 65])
def test_block_load_store_rejects_static_valid_items_outside_tile(
    make_spec,
    valid_items,
):
    with pytest.raises(ValueError, match="block tile size"):
        make_spec(
            dtype="i32",
            block_dim=(16, 2, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(valid_items),
        )


@pytest.mark.parametrize("make_spec", [make_block_load_spec, make_block_store_spec])
@pytest.mark.parametrize("valid_items", [0, 64])
def test_block_load_store_accepts_static_valid_items_at_tile_bounds(
    make_spec,
    valid_items,
):
    spec = make_spec(
        dtype="i32",
        block_dim=(16, 2, 1),
        items_per_thread=2,
        algorithm="direct",
        valid_items=ArgumentBinding.static(valid_items),
    )

    assert spec.has_valid_items


@pytest.mark.parametrize("make_spec", [make_block_load_spec, make_block_store_spec])
def test_block_load_store_canonicalizes_static_valid_items_identity(make_spec):
    def build(valid_items):
        return make_spec(
            dtype="int",
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(valid_items),
        )

    plain = build(5)
    numpy = build(np.int32(5))

    assert plain.call.semantic_key == numpy.call.semantic_key
    assert plain.semantic_key == numpy.semantic_key
    assert numpy.call.valid_items == ArgumentBinding.static(5)


@pytest.mark.parametrize("valid_items", [True, 1.5])
def test_block_load_store_rejects_non_integer_static_valid_items(valid_items):
    with pytest.raises(TypeError, match="must be an integer"):
        make_block_store_spec(
            dtype="i32",
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(valid_items),
        )


@pytest.mark.parametrize("make_spec", [make_block_load_spec, make_block_store_spec])
@pytest.mark.parametrize(
    "algorithm",
    ["warp_transpose", "warp_transpose_timesliced"],
)
def test_direct_block_warp_transpose_requires_complete_physical_warps(
    make_spec,
    algorithm,
):
    with pytest.raises(ValueError, match="multiple of 32"):
        make_spec(
            dtype="i32",
            block_dim=(16, 3, 1),
            items_per_thread=2,
            algorithm=algorithm,
        )


@pytest.mark.parametrize(
    ("value", "literal"),
    [
        (True, "true"),
        (7, "7"),
        (1.5, "1.5"),
        (np.int64(-(1 << 63)), "(-9223372036854775807LL - 1LL)"),
        (np.uint64((1 << 64) - 1), "18446744073709551615ULL"),
    ],
)
def test_block_load_renders_static_oob_default_as_cpp_scalar(value, literal):
    spec = make_block_load_spec(
        dtype="f32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        algorithm="direct",
        valid_items=ArgumentBinding.static(1),
        oob_default=ArgumentBinding.static(value),
    )

    assert spec.specialization.parameters[0][-1].cpp == literal


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_block_load_rejects_nonfinite_static_oob_default(value):
    with pytest.raises(ValueError, match="must be finite"):
        make_block_load_spec(
            dtype="f32",
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(1),
            oob_default=ArgumentBinding.static(value),
        )


@pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 64])
def test_block_load_rejects_static_oob_default_outside_64_bits(value):
    with pytest.raises(ValueError, match="fit a 64-bit integer"):
        make_block_load_spec(
            dtype="i64",
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(1),
            oob_default=ArgumentBinding.static(value),
        )

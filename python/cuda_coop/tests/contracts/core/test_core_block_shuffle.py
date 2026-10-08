# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check scalar and array Shuffle call contracts before backend lowering.

A static distance becomes a C++ expression; a runtime distance remains a
call parameter. The assertions also keep returned scalar values distinct
from arrays written through output parameters.
"""

from inspect import signature

import pytest

from cuda.coop._core import (
    INT32,
    UINT32,
    ArgumentBinding,
    ArgumentKind,
    Array,
    CxxFunction,
    Dependency,
    ParameterRole,
    Reference,
    TempStorageParameter,
    Value,
    classify_parameter,
)
from cuda.coop._core.block import (
    BlockShuffleValueKind,
    make_block_shuffle_semantics,
    make_block_shuffle_specialization,
)


@pytest.mark.parametrize(
    ("mode", "method"),
    [
        ("offset", "Offset"),
        ("rotate", "Rotate"),
        ("up", "Up"),
        ("down", "Down"),
    ],
)
def test_block_shuffle_modes_own_their_cub_method(mode, method):
    semantics = make_block_shuffle_semantics(
        dtype="i32",
        mode=mode,
        distance=ArgumentBinding.static(1),
    )

    assert semantics.method_name == method


def test_scalar_shuffle_preserves_runtime_distance_and_output_shape():
    specialization = make_block_shuffle_specialization(
        dtype="i32",
        block_dim=(16, 2, 1),
        mode="offset",
        distance=ArgumentBinding.runtime(),
    )

    assert specialization.value_kind is BlockShuffleValueKind.SCALAR
    assert specialization.specialization.fake_return
    assert specialization.call.parameters == (
        TempStorageParameter(),
        Reference(Dependency("T"), name="input_item"),
        Reference(Dependency("T"), name="output_item", is_output=True),
        Value(INT32, name="distance"),
    )
    assert [
        entry.role
        for entry in map(
            classify_parameter, specialization.specialization.parameters[0]
        )
    ] == [
        ParameterRole.TEMP_STORAGE,
        ParameterRole.INPUT,
        ParameterRole.OUTPUT,
        ParameterRole.INPUT,
    ]
    assert all(
        entry.kind is ArgumentKind.RUNTIME
        for entry in map(
            classify_parameter, specialization.specialization.parameters[0]
        )
    )

    rotate = make_block_shuffle_specialization(
        dtype="i32",
        block_dim=(16, 2, 1),
        mode="rotate",
        distance=ArgumentBinding.runtime(),
    )
    assert rotate.call.parameters[-1] == Value(UINT32, name="distance")


def test_scalar_static_and_default_distances_are_core_constants():
    offset = make_block_shuffle_specialization(
        dtype="i32",
        block_dim=(32, 1, 1),
        mode="offset",
        distance=ArgumentBinding.static(-3),
    )
    rotate = make_block_shuffle_specialization(
        dtype="i32",
        block_dim=(32, 1, 1),
        mode="rotate",
    )

    assert offset.call.parameters[-1] == CxxFunction(
        "-3", INT32, name="distance"
    )
    assert rotate.call.parameters[-1] == CxxFunction(
        "1", UINT32, name="distance"
    )
    assert (
        classify_parameter(offset.specialization.parameters[0][-1]).kind
        is ArgumentKind.STATIC
    )
    assert (
        classify_parameter(rotate.specialization.parameters[0][-1]).kind
        is ArgumentKind.STATIC
    )


@pytest.mark.parametrize("mode", ["up", "down"])
def test_array_shuffle_is_out_of_place_without_boundary_outputs(mode):
    specialization = make_block_shuffle_specialization(
        dtype="i64",
        block_dim=(8, 4, 2),
        mode=mode,
        items_per_thread=3,
    )

    assert specialization.value_kind is BlockShuffleValueKind.ARRAY
    assert (
        specialization.specialization.template_arguments["ITEMS_PER_THREAD"]
        == 3
    )
    assert specialization.call.parameters == (
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
    )
    names = signature(make_block_shuffle_semantics).parameters
    assert "block_prefix" not in names
    assert "block_suffix" not in names


def test_block_shuffle_identity_tracks_shape_mode_and_distance_policy():
    def make(**overrides):
        options = {
            "dtype": "i32",
            "mode": "rotate",
            "distance": ArgumentBinding.static(1),
        }
        options.update(overrides)
        return make_block_shuffle_semantics(**options)

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(dtype="i64").semantic_key
    assert make().semantic_key != make(mode="offset").semantic_key
    assert (
        make().semantic_key
        != make(distance=ArgumentBinding.static(2)).semantic_key
    )
    assert (
        make().semantic_key
        != make(distance=ArgumentBinding.runtime()).semantic_key
    )
    assert make().semantic_key != make(items_per_thread=2).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_block_shuffle_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="up",
            items_per_thread=items_per_thread,
        )


def test_block_shuffle_rejects_invalid_public_cub_shapes():
    with pytest.raises(TypeError, match="static distance must be an integer"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="offset",
            distance=ArgumentBinding.static(True),
        )
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="rotate",
            distance=ArgumentBinding.static(-1),
        )
    for distance in (0, 32):
        with pytest.raises(ValueError, match="1 <= distance < block_threads"):
            make_block_shuffle_specialization(
                dtype="i32",
                block_dim=(32, 1, 1),
                mode="rotate",
                distance=ArgumentBinding.static(distance),
            )
    with pytest.raises(ValueError, match="1 <= distance < block_threads"):
        make_block_shuffle_specialization(
            dtype="i32",
            block_dim=(1, 1, 1),
            mode="rotate",
        )
    with pytest.raises(ValueError, match="scalar BlockShuffle"):
        make_block_shuffle_specialization(
            dtype="i32",
            block_dim=(32, 1, 1),
            mode="up",
        )
    with pytest.raises(ValueError, match="array BlockShuffle"):
        make_block_shuffle_specialization(
            dtype="i32",
            block_dim=(32, 1, 1),
            mode="rotate",
            items_per_thread=2,
        )
    with pytest.raises(ValueError, match="does not accept distance"):
        make_block_shuffle_specialization(
            dtype="i32",
            block_dim=(32, 1, 1),
            mode="down",
            items_per_thread=2,
            distance=ArgumentBinding.static(1),
        )

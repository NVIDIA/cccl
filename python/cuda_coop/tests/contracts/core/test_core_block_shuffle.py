# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    INT32,
    ArgumentBinding,
    ArgumentKind,
    Array,
    CxxFunction,
    Dependency,
    ParameterRole,
    Pointer,
    Reference,
    TempStorageParameter,
    Value,
)
from cuda.coop._core.block import (
    BlockShuffleMode,
    BlockShuffleValueKind,
    make_block_shuffle_semantics,
    make_block_shuffle_spec,
)


@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        ("offset", "Offset"),
        ("rotate", "Rotate"),
        ("up", "Up"),
        ("down", "Down"),
    ],
)
def test_block_shuffle_mode_owns_cub_method_mapping(mode, method_name):
    call = make_block_shuffle_semantics(
        dtype="i32",
        mode=mode,
        distance=ArgumentBinding.static(1),
    )

    assert call.mode.cub_method_name == method_name
    assert BlockShuffleMode.from_cub_method_name(method_name) is call.mode


def test_scalar_block_shuffle_preserves_runtime_distance_and_return_shape():
    spec = make_block_shuffle_spec(
        dtype="i32",
        block_dim=(16, 2, 1),
        mode="offset",
        distance=ArgumentBinding.runtime(),
    )

    assert spec.value_kind is BlockShuffleValueKind.SCALAR
    assert spec.method_name == "Offset"
    assert spec.specialization.fake_return
    assert spec.call.parameters == (
        TempStorageParameter(),
        Reference(Dependency("T"), name="input_item"),
        Reference(
            Dependency("T"),
            name="output_item",
            is_output=True,
        ),
        Value(INT32, name="distance"),
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
    ]


def test_scalar_static_distance_becomes_a_core_constant():
    spec = make_block_shuffle_spec(
        dtype="i32",
        block_dim=(32, 1, 1),
        mode="rotate",
        distance=ArgumentBinding.static(3),
    )

    assert spec.call.parameters[-1] == CxxFunction("3", INT32, name="distance")
    assert spec.specialization.classify_method()[-1].kind is ArgumentKind.STATIC


def test_array_up_suffix_preserves_parameter_order_and_auxiliary_item_count():
    spec = make_block_shuffle_spec(
        dtype="i32",
        block_dim=(8, 4, 2),
        mode="up",
        items_per_thread=3,
        block_suffix=True,
    )

    assert spec.value_kind is BlockShuffleValueKind.ARRAY
    assert spec.call.parameters == (
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
        Pointer(
            Dependency("T"),
            name="block_suffix",
            is_output=True,
            is_return=False,
            is_array_pointer=True,
            deref_on_call=True,
        ),
    )
    assert spec.specialization.template_arguments["ITEMS_PER_THREAD"] == 3
    assert spec.specialization.ordered_specialization_arguments[-1] == (
        "ITEMS_PER_THREAD",
        3,
    )


def test_dimension_independent_contract_allows_cute_extended_array_modes():
    call = make_block_shuffle_semantics(
        dtype="i32",
        mode="rotate",
        items_per_thread=4,
        distance=ArgumentBinding.static(2),
    )

    assert call.is_array
    assert call.parameters[-1] == CxxFunction("2", INT32, name="distance")
    with pytest.raises(ValueError, match="array BlockShuffle supports only"):
        make_block_shuffle_spec(
            dtype="i32",
            block_dim=(32, 1, 1),
            mode="rotate",
            items_per_thread=4,
            distance=ArgumentBinding.static(2),
        )

    scalar_with_prefix = make_block_shuffle_semantics(
        dtype="i32",
        mode="offset",
        distance=ArgumentBinding.static(2),
        block_prefix=True,
    )
    assert scalar_with_prefix.parameters[-1].name == "block_prefix"
    assert scalar_with_prefix.parameters[-1].role is ParameterRole.OUTPUT


def test_block_shuffle_semantic_identity_tracks_shape_and_binding_policy():
    def make(**kwargs):
        options = {
            "dtype": "i32",
            "mode": "down",
            "items_per_thread": 2,
            "distance": ArgumentBinding.static(1),
            "block_prefix": True,
        }
        options.update(kwargs)
        return make_block_shuffle_semantics(**options)

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(items_per_thread=3).semantic_key
    assert make().semantic_key != make(distance=ArgumentBinding.runtime()).semantic_key
    assert make().semantic_key != make(block_prefix=False).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_block_shuffle_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="up",
            items_per_thread=items_per_thread,
        )


def test_block_shuffle_rejects_invalid_mode_shape_and_boundary_combinations():
    with pytest.raises(ValueError, match="static distance"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="offset",
            distance=ArgumentBinding.static(True),
        )
    with pytest.raises(ValueError, match="non-negative"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="up",
            distance=ArgumentBinding.static(-1),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="up",
            block_prefix=True,
            block_suffix=True,
        )
    with pytest.raises(ValueError, match="block_prefix"):
        make_block_shuffle_semantics(
            dtype="i32",
            mode="rotate",
            block_prefix=True,
        )
    with pytest.raises(ValueError, match="scalar BlockShuffle"):
        make_block_shuffle_spec(
            dtype="i32",
            block_dim=(32, 1, 1),
            mode="up",
        )
    with pytest.raises(ValueError, match="does not accept distance"):
        make_block_shuffle_spec(
            dtype="i32",
            block_dim=(32, 1, 1),
            mode="down",
            items_per_thread=2,
            distance=ArgumentBinding.static(1),
        )

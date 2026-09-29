# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from cuda.coop._core import ArgumentBinding, ArgumentKind
from cuda.coop._core.block import (
    make_block_load_spec,
    make_block_load_store_semantics,
    make_block_store_spec,
)


def test_block_load_partial_default_and_pointer_offset_overloads():
    spec = make_block_load_spec(
        dtype="f32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        algorithm="transpose",
        valid_items=True,
        oob_default=True,
        include_full_tile=True,
        include_pointer_offset=True,
    )

    # Each optional overload must preserve CUB's argument order.
    assert [
        [parameter.name for parameter in method[3:]]
        for method in spec.specialization.parameters
    ] == [
        [],
        ["num_valid_items", "oob_default"],
        ["num_valid_items", "oob_default", "offset"],
        ["offset"],
    ]


def test_block_load_static_controls_stay_out_of_the_runtime_abi():
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
    controls = spec.specialization.parameters[0][-3:]
    assert all(p.argument_kind is ArgumentKind.STATIC for p in controls)
    assert controls[-1].static_value == 4


@pytest.mark.parametrize(
    "make_spec", [make_block_load_spec, make_block_store_spec]
)
def test_numpy_integer_controls_share_the_same_specialization(make_spec):
    def build(integer):
        return make_spec(
            dtype="i32",
            block_dim=(32, 1, 1),
            items_per_thread=integer(2),
            algorithm="direct",
            valid_items=ArgumentBinding.static(integer(5)),
            include_pointer_offset=ArgumentBinding.static(integer(7)),
        )

    assert build(int).semantic_key == build(np.int64).semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, "two"])
def test_block_load_store_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(
        ValueError, match="items_per_thread must be a positive integer"
    ):
        make_block_load_store_semantics(
            kind="load",
            dtype="i32",
            items_per_thread=items_per_thread,
            algorithm="direct",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"algorithm": "unknown"}, "unsupported BlockLoad algorithm"),
        (
            {"kind": "store", "valid_items": True, "oob_default": True},
            "only valid for BlockLoad",
        ),
        ({"oob_default": True}, "requires a valid_items"),
        ({"include_full_tile": True}, "include_full_tile"),
    ],
)
def test_block_load_store_rejects_invalid_options(overrides, message):
    options = {
        "kind": "load",
        "dtype": "i32",
        "items_per_thread": 1,
        "algorithm": "direct",
    }
    options.update(overrides)
    with pytest.raises(ValueError, match=message):
        make_block_load_store_semantics(**options)


@pytest.mark.parametrize(
    "make_spec", [make_block_load_spec, make_block_store_spec]
)
def test_block_valid_items_bounds_use_all_three_dimensions(make_spec):
    def build(value):
        return make_spec(
            dtype="i32",
            block_dim=(8, 2, 2),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(value),
        )

    for value in (0, 64):
        build(value)
    for value in (-1, 65):
        with pytest.raises(ValueError, match="block tile size"):
            build(value)


@pytest.mark.parametrize(
    "make_spec", [make_block_load_spec, make_block_store_spec]
)
@pytest.mark.parametrize(
    "algorithm", ["warp_transpose", "warp_transpose_timesliced"]
)
def test_direct_block_warp_transpose_requires_complete_physical_warps(
    make_spec, algorithm
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


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("inf"), "must be finite"),
        (float("-inf"), "must be finite"),
        (float("nan"), "must be finite"),
        (-(1 << 63) - 1, "fit a 64-bit integer"),
        (1 << 64, "fit a 64-bit integer"),
    ],
)
def test_block_load_rejects_unrepresentable_static_oob_default(value, message):
    with pytest.raises(ValueError, match=message):
        make_block_load_spec(
            dtype="f32",
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(1),
            oob_default=ArgumentBinding.static(value),
        )

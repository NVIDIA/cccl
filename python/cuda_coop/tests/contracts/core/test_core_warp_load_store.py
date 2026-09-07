# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import (
    INT32,
    ArgumentBinding,
    ArgumentKind,
    Array,
    Dependency,
    ParameterRole,
    Pointer,
    TempStorageParameter,
    Value,
)
from cuda.coop._core.warp import (
    WarpLoadAlgorithm,
    WarpLoadStoreAlgorithm,
    WarpLoadStoreKind,
    WarpStoreAlgorithm,
    make_warp_load_spec,
    make_warp_store_spec,
)


def test_warp_load_partial_oob_semantics():
    spec = make_warp_load_spec(
        dtype="i32",
        items_per_thread=4,
        threads_in_warp=16,
        algorithm="transpose",
        valid_items=True,
        oob_default=True,
    )

    assert spec.kind is WarpLoadStoreKind.LOAD
    assert WarpLoadAlgorithm is WarpLoadStoreAlgorithm
    assert WarpStoreAlgorithm is WarpLoadStoreAlgorithm
    assert spec.algorithm is WarpLoadAlgorithm.TRANSPOSE
    assert spec.algorithm_cpp == "::cub::WARP_LOAD_TRANSPOSE"
    assert spec.method_name == "Load"
    assert spec.has_valid_items
    assert spec.has_oob_default
    assert not spec.has_full_tile
    assert spec.specialization.template_arguments == {
        "T": "i32",
        "ITEMS_PER_THREAD": 4,
        "ALGORITHM": "::cub::WARP_LOAD_TRANSPOSE",
        "LOGICAL_WARP_THREADS": 16,
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
            Value(INT32, name="num_valid_items"),
            Value("i32", name="oob_default"),
        ),
    )
    classifications = spec.specialization.classify_method()
    assert [(item.kind, item.role) for item in classifications] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
    ]


def test_warp_store_full_tile_semantics_accept_cub_spelling():
    spec = make_warp_store_spec(
        dtype="i64",
        items_per_thread=2,
        threads_in_warp=32,
        algorithm="::cub::WARP_STORE_STRIPED",
    )

    assert spec.kind is WarpLoadStoreKind.STORE
    assert spec.algorithm is WarpStoreAlgorithm.STRIPED
    assert spec.algorithm_cpp == "::cub::WARP_STORE_STRIPED"
    assert spec.has_full_tile
    assert not spec.has_valid_items
    assert [parameter.name for parameter in spec.specialization.parameters[0]] == [
        "temp_storage",
        "dst",
        "src",
    ]
    assert spec.specialization.parameters[0][1].is_output
    assert spec.specialization.parameters[0][1].is_return is False


def test_warp_load_can_retain_full_and_partial_overloads():
    spec = make_warp_load_spec(
        dtype="i32",
        items_per_thread=1,
        threads_in_warp=8,
        algorithm=WarpLoadAlgorithm.DIRECT,
        valid_items=True,
        include_full_tile=True,
    )

    assert spec.has_full_tile
    assert [
        [item.name for item in method] for method in spec.specialization.parameters
    ] == [
        ["temp_storage", "src", "dst"],
        ["temp_storage", "src", "dst", "num_valid_items"],
    ]


@pytest.mark.parametrize(
    ("make_spec", "pointer_name", "value_name"),
    [
        (make_warp_load_spec, "src", "dst"),
        (make_warp_store_spec, "dst", "src"),
    ],
)
def test_warp_load_store_orders_full_partial_and_pointer_offset_overloads(
    make_spec,
    pointer_name,
    value_name,
):
    spec = make_spec(
        dtype="i32",
        items_per_thread=2,
        threads_in_warp=16,
        algorithm="direct",
        valid_items=True,
        include_full_tile=True,
        include_pointer_offset=True,
    )

    base = ["temp_storage", pointer_name, value_name]
    assert [
        [parameter.name for parameter in method]
        for method in spec.specialization.parameters
    ] == [
        base,
        [*base, "num_valid_items"],
        [*base, "num_valid_items", "offset"],
        [*base, "offset"],
    ]


def test_warp_pointer_offset_identity_normalizes_numpy_integer():
    kwargs = {
        "dtype": "i32",
        "items_per_thread": 2,
        "threads_in_warp": 16,
        "algorithm": "direct",
    }
    plain = make_warp_load_spec(
        **kwargs,
        include_pointer_offset=ArgumentBinding.static(4),
    )
    numpy = make_warp_load_spec(
        **kwargs,
        include_pointer_offset=ArgumentBinding.static(np.int64(4)),
    )

    assert numpy.specialization.metadata["pointer_offset"] == (
        "static",
        "builtins",
        "int",
        "4",
    )
    assert plain.semantic_key == numpy.semantic_key


def test_warp_load_store_semantic_identity_tracks_operation_shape():
    def load(*, algorithm="direct", valid=False):
        return make_warp_load_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=16,
            algorithm=algorithm,
            valid_items=valid,
        )

    assert load().semantic_key == load().semantic_key
    assert load().semantic_key != load(algorithm="striped").semantic_key
    assert load().semantic_key != load(valid=True).semantic_key
    assert (
        load().semantic_key
        != make_warp_store_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=16,
            algorithm="direct",
        ).semantic_key
    )


@pytest.mark.parametrize("items_per_thread", [True, 0, 1.5])
def test_warp_load_rejects_invalid_items_per_thread(items_per_thread):
    with pytest.raises(ValueError, match="positive integer"):
        make_warp_load_spec(
            dtype="i32",
            items_per_thread=items_per_thread,
            threads_in_warp=32,
            algorithm="direct",
        )


def test_warp_load_store_reject_invalid_option_combinations():
    with pytest.raises(ValueError, match="requires a valid_items"):
        make_warp_load_spec(
            dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            algorithm="direct",
            oob_default=True,
        )
    with pytest.raises(ValueError, match="only valid for WarpLoad"):
        make_warp_store_spec(
            dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            algorithm="direct",
            valid_items=True,
            oob_default=True,
        )
    with pytest.raises(ValueError, match="include_full_tile requires"):
        make_warp_load_spec(
            dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            algorithm="direct",
            include_full_tile=True,
        )
    with pytest.raises(ValueError, match="unsupported WarpLoad algorithm"):
        make_warp_load_spec(
            dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            algorithm="warp_transpose",
        )


@pytest.mark.parametrize("make_spec", [make_warp_load_spec, make_warp_store_spec])
@pytest.mark.parametrize("valid_items", [-1, 65])
def test_warp_load_store_rejects_static_valid_items_outside_tile(
    make_spec,
    valid_items,
):
    with pytest.raises(ValueError, match="warp tile size"):
        make_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=32,
            algorithm="direct",
            valid_items=ArgumentBinding.static(valid_items),
        )


@pytest.mark.parametrize("make_spec", [make_warp_load_spec, make_warp_store_spec])
@pytest.mark.parametrize("valid_items", [0, 64])
def test_warp_load_store_accepts_static_valid_items_at_tile_bounds(
    make_spec,
    valid_items,
):
    spec = make_spec(
        dtype="i32",
        items_per_thread=2,
        threads_in_warp=32,
        algorithm="direct",
        valid_items=ArgumentBinding.static(valid_items),
    )

    assert spec.has_valid_items


def test_warp_load_store_rejects_non_integer_static_valid_items():
    with pytest.raises(TypeError, match="must be an integer"):
        make_warp_store_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=32,
            algorithm="direct",
            valid_items=ArgumentBinding.static(1.5),
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
def test_warp_load_renders_static_oob_default_as_cpp_scalar(value, literal):
    spec = make_warp_load_spec(
        dtype="f32",
        items_per_thread=2,
        threads_in_warp=32,
        algorithm="direct",
        valid_items=ArgumentBinding.static(1),
        oob_default=ArgumentBinding.static(value),
    )

    assert spec.specialization.parameters[0][-1].cpp == literal


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_warp_load_rejects_nonfinite_static_oob_default(value):
    with pytest.raises(ValueError, match="must be finite"):
        make_warp_load_spec(
            dtype="f32",
            items_per_thread=2,
            threads_in_warp=32,
            algorithm="direct",
            valid_items=ArgumentBinding.static(1),
            oob_default=ArgumentBinding.static(value),
        )

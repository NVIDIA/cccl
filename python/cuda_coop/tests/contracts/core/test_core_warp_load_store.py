# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    INT32,
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

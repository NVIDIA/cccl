# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


from types import SimpleNamespace

import pytest

from cuda.coop._core import (
    ArgumentKind,
    Array,
    Dependency,
    ParameterRole,
    TempStorageParameter,
)
from cuda.coop._core.block import (
    BlockHistogramAlgorithm,
    BlockHistogramOperation,
    make_block_histogram_semantics,
    make_block_histogram_spec,
    normalize_block_histogram_algorithm,
    validate_block_histogram_output_capacity,
)


def test_histogram_spec_owns_cub_specialization_and_full_operation_abi():
    spec = make_block_histogram_spec(
        item_dtype="u8",
        counter_dtype="u32",
        block_dim=(16, 2, 1),
        items_per_thread=4,
        bins=256,
        algorithm="atomic",
    )

    assert spec.operation is BlockHistogramOperation.HISTOGRAM
    assert spec.algorithm is BlockHistogramAlgorithm.ATOMIC
    assert spec.algorithm_cpp == "::cub::BLOCK_HISTO_ATOMIC"
    assert spec.method_name == "Histogram"
    assert spec.specialization.c_name == "block_histogram"
    assert spec.block_dim == (16, 2, 1)
    assert spec.specialization.template_arguments == {
        "T": "u8",
        "BLOCK_DIM_X": 16,
        "ITEMS_PER_THREAD": 4,
        "BINS": 256,
        "ALGORITHM": "::cub::BLOCK_HISTO_ATOMIC",
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="items",
            ),
            Array(
                "u32",
                Dependency("BINS"),
                name="histogram",
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


def test_init_and_composite_preserve_distinct_cub_parameter_roles():
    init = make_block_histogram_spec(
        item_dtype="i32",
        counter_dtype="i64",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        bins=64,
        algorithm="sort",
        operation="init",
    )
    composite = make_block_histogram_spec(
        item_dtype="i32",
        counter_dtype="i64",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        bins=64,
        algorithm="::cub::BLOCK_HISTO_SORT",
        operation="composite",
    )

    assert init.method_name == "InitHistogram"
    assert init.specialization.c_name == "block_histogram_init"
    assert [parameter.name for parameter in init.specialization.parameters[0]] == [
        "temp_storage",
        "histogram",
    ]
    assert init.specialization.classify_method()[1].role is ParameterRole.OUTPUT

    assert composite.method_name == "Composite"
    assert composite.specialization.c_name == "block_histogram_composite"
    assert [parameter.name for parameter in composite.specialization.parameters[0]] == [
        "temp_storage",
        "items",
        "histogram",
    ]
    assert composite.specialization.classify_method()[2].role is ParameterRole.INOUT
    assert init.semantic_key != composite.semantic_key


def test_instance_semantics_carry_only_parent_storage_identity():
    spec = make_block_histogram_spec(
        item_dtype="u8",
        counter_dtype="u32",
        block_dim=(128, 1, 1),
        items_per_thread=4,
        bins=256,
        algorithm="sort",
        operation="instance",
    )

    assert spec.operation is BlockHistogramOperation.INSTANCE
    assert spec.method_name == "Histogram"
    assert spec.specialization.c_name == "block_histogram"
    assert spec.specialization.parameters == ((TempStorageParameter(),),)


def test_runtime_bin_semantics_support_provider_parity_without_cub_specialization():
    call = make_block_histogram_semantics(
        item_dtype="i32",
        counter_dtype="u64",
        items_per_thread=3,
        bins=None,
        algorithm="block_histo_atomic",
        operation=BlockHistogramOperation.HISTOGRAM,
    )

    assert not call.has_static_bins
    assert call.algorithm is BlockHistogramAlgorithm.ATOMIC
    assert call.method_name == "Histogram"
    assert [parameter.name for parameter in call.parameters] == [
        "temp_storage",
        "items",
        "histogram",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, BlockHistogramAlgorithm.ATOMIC),
        ("atomic", BlockHistogramAlgorithm.ATOMIC),
        ("BLOCK_HISTO_ATOMIC", BlockHistogramAlgorithm.ATOMIC),
        ("::cub::BLOCK_HISTO_SORT", BlockHistogramAlgorithm.SORT),
        ("BlockHistogramAlgorithm.SORT", BlockHistogramAlgorithm.SORT),
        (SimpleNamespace(name="ATOMIC"), BlockHistogramAlgorithm.ATOMIC),
    ],
)
def test_algorithm_normalization_accepts_frontend_and_cub_spellings(value, expected):
    assert normalize_block_histogram_algorithm(value) is expected


def test_histogram_semantic_identity_tracks_operation_shape_algorithm_and_types():
    def make(**overrides):
        options = {
            "item_dtype": "u8",
            "counter_dtype": "u32",
            "items_per_thread": 4,
            "bins": 256,
            "algorithm": "atomic",
            "operation": "histogram",
        }
        options.update(overrides)
        return make_block_histogram_semantics(**options)

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(operation="composite").semantic_key
    assert make().semantic_key != make(algorithm="sort").semantic_key
    assert make().semantic_key != make(items_per_thread=2).semantic_key
    assert make().semantic_key != make(bins=128).semantic_key
    assert make().semantic_key != make(counter_dtype="u64").semantic_key


@pytest.mark.parametrize("items_per_thread", [0, -1, True, 1.5, "two"])
def test_histogram_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        make_block_histogram_semantics(
            item_dtype="u8",
            counter_dtype="u32",
            items_per_thread=items_per_thread,
            bins=256,
        )


@pytest.mark.parametrize("bins", [0, -1, True, 1.5, "many"])
def test_histogram_rejects_invalid_static_bin_count(bins):
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        make_block_histogram_semantics(
            item_dtype="u8",
            counter_dtype="u32",
            items_per_thread=4,
            bins=bins,
        )


def test_histogram_rejects_missing_types_algorithm_and_block_shape():
    options = {
        "item_dtype": "u8",
        "counter_dtype": "u32",
        "items_per_thread": 4,
        "bins": 256,
    }
    with pytest.raises(ValueError, match="item dtype must be provided"):
        make_block_histogram_semantics(**{**options, "item_dtype": None})
    with pytest.raises(ValueError, match="counter dtype must be provided"):
        make_block_histogram_semantics(**{**options, "counter_dtype": None})
    with pytest.raises(ValueError, match="unsupported BlockHistogram algorithm"):
        make_block_histogram_semantics(**options, algorithm="bogus")
    with pytest.raises(ValueError, match="block_dim"):
        make_block_histogram_spec(
            **options,
            block_dim=(32, 0, 1),
        )


def test_histogram_output_capacity_covers_every_striped_bin():
    validate_block_histogram_output_capacity(
        bins=64,
        bins_per_thread=1,
        block_threads=64,
    )
    validate_block_histogram_output_capacity(
        bins=65,
        bins_per_thread=2,
        block_threads=64,
    )

    with pytest.raises(
        TypeError,
        match="histogram bins must be a compile-time positive integer",
    ):
        validate_block_histogram_output_capacity(
            bins="65",
            bins_per_thread=2,
            block_threads=64,
        )

    with pytest.raises(
        ValueError,
        match=("histogram bins_per_thread must be a compile-time positive integer"),
    ):
        validate_block_histogram_output_capacity(
            bins=65,
            bins_per_thread=0,
            block_threads=64,
        )

    with pytest.raises(
        ValueError,
        match=(
            "histogram bins_per_thread is too small for 65 bins and block "
            "size 64; need at least 2"
        ),
    ):
        validate_block_histogram_output_capacity(
            bins=65,
            bins_per_thread=1,
            block_threads=64,
        )

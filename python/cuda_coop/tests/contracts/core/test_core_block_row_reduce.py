# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import ArgumentKind, ParameterRole
from cuda.coop._core.block import (
    BLOCK_ROW_REDUCE_INCLUDES,
    make_block_row_reduce_spec,
    normalize_block_row_reduce_geometry,
)


def test_block_row_reduce_owns_exact_public_cub_specialization_and_abi():
    spec = make_block_row_reduce_spec(
        dtype="float",
        rows_per_block=2,
        warps_per_row=4,
    )

    algorithm = spec.specialization
    assert algorithm.struct_name == "BlockRowReduceWarpBroadcast"
    assert algorithm.method_name == "Sum"
    assert algorithm.c_name == "block_row_reduce"
    assert algorithm.includes == BLOCK_ROW_REDUCE_INCLUDES
    assert algorithm.template_parameter_names == (
        "T",
        "ROWS_PER_BLOCK",
        "WARPS_PER_ROW",
    )
    assert algorithm.ordered_template_arguments == (
        ("T", "float"),
        ("ROWS_PER_BLOCK", 2),
        ("WARPS_PER_ROW", 4),
    )
    assert [
        (parameter.name, parameter.kind, parameter.role)
        for parameter in algorithm.classify_method()
    ] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("value", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]
    assert algorithm.parameters[0][-1].is_return is True
    assert spec.logical_warps == 8
    assert spec.block_threads == 256


def test_block_row_reduce_identity_tracks_dtype_and_static_geometry():
    def make(dtype="float", rows=2, warps=4):
        return make_block_row_reduce_spec(
            dtype=dtype,
            rows_per_block=rows,
            warps_per_row=warps,
        )

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(dtype="double").semantic_key
    assert make().semantic_key != make(rows=1).semantic_key
    assert make().semantic_key != make(warps=8).semantic_key


def test_block_row_reduce_normalizes_non_boolean_integral_geometry():
    geometry = normalize_block_row_reduce_geometry(
        rows_per_block=np.int64(2),
        warps_per_row=np.int32(4),
    )

    assert type(geometry.rows_per_block) is int
    assert type(geometry.warps_per_row) is int
    assert geometry.rows_per_block == 2
    assert geometry.warps_per_row == 4
    assert geometry.logical_warps == 8
    assert geometry.block_threads == 256


def test_block_row_reduce_geometry_requires_exact_block_width():
    geometry = normalize_block_row_reduce_geometry(
        rows_per_block=1,
        warps_per_row=4,
    )

    geometry.validate_block_threads(128)
    with pytest.raises(ValueError, match="block has 64 threads; expected exactly 128"):
        geometry.validate_block_threads(64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rows_per_block", True, "rows_per_block must be a positive integer"),
        ("rows_per_block", 0, "rows_per_block must be a positive integer"),
        ("rows_per_block", 1.5, "rows_per_block must be a positive integer"),
        ("warps_per_row", 0, "warps_per_row must be a positive integer"),
        ("warps_per_row", 33, "warps_per_row must be <= 32"),
        (
            "rows_per_block",
            2,
            "rows_per_block \\* warps_per_row must fit in one CUDA thread block",
        ),
    ],
)
def test_block_row_reduce_rejects_invalid_geometry(field, value, message):
    kwargs = {"rows_per_block": 1, "warps_per_row": 17}
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        normalize_block_row_reduce_geometry(**kwargs)


def test_block_row_reduce_requires_dtype_without_probing_header_availability():
    with pytest.raises(ValueError, match="dtype must be provided"):
        make_block_row_reduce_spec(
            dtype=None,
            rows_per_block=1,
            warps_per_row=4,
        )

    # Header availability is a backend/compiler concern. Core can describe the
    # optional newer-CUB contract without importing or probing a toolkit.
    assert make_block_row_reduce_spec(
        dtype="float",
        rows_per_block=1,
        warps_per_row=4,
    ).specialization.includes == ("cub/block/block_row_reduce.cuh",)

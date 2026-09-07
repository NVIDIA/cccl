# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core.dtype_policy import (
    COMMON_V1_INTEGER_KEY_DTYPE_NAMES,
    COMMON_V1_INTEGER_VALUE_DTYPE_NAMES,
    COMMON_V1_NUMERIC_DTYPE_NAMES,
    validate_common_v1_integer_key_dtype_name,
    validate_common_v1_integer_value_dtype_name,
    validate_common_v1_numeric_dtype_name,
)


@pytest.mark.parametrize(
    ("dtype_names", "validator", "parameter"),
    [
        (COMMON_V1_NUMERIC_DTYPE_NAMES, validate_common_v1_numeric_dtype_name, None),
        (
            COMMON_V1_INTEGER_VALUE_DTYPE_NAMES,
            validate_common_v1_integer_value_dtype_name,
            "sample",
        ),
        (
            COMMON_V1_INTEGER_KEY_DTYPE_NAMES,
            validate_common_v1_integer_key_dtype_name,
            "key",
        ),
    ],
)
def test_common_dtype_policy_accepts_each_certified_family(
    dtype_names,
    validator,
    parameter,
) -> None:
    kwargs = {"operation": "histogram"}
    if parameter is not None:
        kwargs["parameter"] = parameter
    for dtype_name in dtype_names:
        assert validator(dtype_name, **kwargs) == dtype_name


@pytest.mark.parametrize(
    "dtype_name",
    ["bool", "int8", "int16", "uint16", "float16", "complex64", "complex128"],
)
def test_common_numeric_dtype_policy_rejects_backend_extensions(dtype_name) -> None:
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.exchange common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        validate_common_v1_numeric_dtype_name(dtype_name, operation="exchange")


@pytest.mark.parametrize(
    ("validator", "dtype_name", "parameter", "supported"),
    [
        (
            validate_common_v1_integer_value_dtype_name,
            "float32",
            "sample",
            "uint8, int32, uint32, int64, uint64",
        ),
        (
            validate_common_v1_integer_key_dtype_name,
            "uint8",
            "counter",
            "int32, uint32, int64, uint64",
        ),
    ],
)
def test_common_integer_dtype_diagnostics_are_backend_neutral(
    validator,
    dtype_name,
    parameter,
    supported,
) -> None:
    expected = (
        f"cuda.coop.histogram common V1 supports {parameter} dtypes {supported}; "
        f"use a backend-qualified import for backend-specific {parameter} dtypes"
    )
    with pytest.raises(TypeError) as exc_info:
        validator(
            dtype_name,
            operation="histogram",
            parameter=parameter,
        )
    assert str(exc_info.value) == expected

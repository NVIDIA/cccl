# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light dtype policy for the certified common V1 profile."""

from __future__ import annotations

COMMON_V1_NUMERIC_DTYPE_NAMES = (
    "uint8",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
)

COMMON_V1_INTEGER_VALUE_DTYPE_NAMES = (
    "uint8",
    "int32",
    "uint32",
    "int64",
    "uint64",
)

COMMON_V1_INTEGER_KEY_DTYPE_NAMES = (
    "int32",
    "uint32",
    "int64",
    "uint64",
)


def _validate_common_v1_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str | None,
    supported_dtype_names: tuple[str, ...],
) -> str:
    """Validate one normalized dtype name and report the common contract."""

    if dtype_name not in supported_dtype_names:
        supported = ", ".join(supported_dtype_names)
        subject = "dtypes" if parameter is None else f"{parameter} dtypes"
        raise TypeError(
            f"cuda.coop.{operation} common V1 supports {subject} {supported}; "
            f"use a backend-qualified import for backend-specific {subject}"
        )
    return dtype_name


def validate_common_v1_numeric_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str | None = None,
) -> str:
    """Validate one backend-normalized dtype name for a common operation."""

    return _validate_common_v1_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
        supported_dtype_names=COMMON_V1_NUMERIC_DTYPE_NAMES,
    )


def validate_common_v1_integer_value_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str = "value",
) -> str:
    """Validate one normalized dtype name for a common integer value."""

    return _validate_common_v1_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
        supported_dtype_names=COMMON_V1_INTEGER_VALUE_DTYPE_NAMES,
    )


def validate_common_v1_integer_key_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str = "key",
) -> str:
    """Validate one normalized dtype name for a common integer key."""

    return _validate_common_v1_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
        supported_dtype_names=COMMON_V1_INTEGER_KEY_DTYPE_NAMES,
    )

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light dtype policy for the portable root API."""

from __future__ import annotations

_PORTABLE_NUMERIC_DTYPE_NAMES = (
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
)

_PORTABLE_INTEGER_VALUE_DTYPE_NAMES = (
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
)


def _validate_portable_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str | None,
    supported_dtype_names: tuple[str, ...],
) -> str:
    """Validate one normalized dtype name and report the portable contract."""

    if dtype_name not in supported_dtype_names:
        supported = ", ".join(supported_dtype_names)
        subject = "dtypes" if parameter is None else f"{parameter} dtypes"
        raise TypeError(
            f"cuda.coop.{operation} supports {subject} {supported} through the "
            "portable API; "
            f"use a backend-qualified import for backend-specific {subject}"
        )
    return dtype_name


def validate_portable_numeric_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str | None = None,
) -> str:
    """Validate one backend-normalized dtype name for a portable operation."""

    return _validate_portable_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
        supported_dtype_names=_PORTABLE_NUMERIC_DTYPE_NAMES,
    )


def validate_portable_integer_value_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str = "value",
) -> str:
    """Validate one normalized dtype name for a portable integer value."""

    return _validate_portable_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
        supported_dtype_names=_PORTABLE_INTEGER_VALUE_DTYPE_NAMES,
    )

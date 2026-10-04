# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keep the common API's numeric type choices consistent across backends.

Callers first convert a dtype object to a standard name such as ``int32``.
Common payload checks and backends both do this. These checks then enforce
the common operation's supported set. Names keep this policy independent
of each compiler's type objects.
"""

from __future__ import annotations

_COMMON_NUMERIC_DTYPE_NAMES = (
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

_COMMON_INTEGER_VALUE_DTYPE_NAMES = (
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
)


def _validate_common_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str | None,
    supported_dtype_names: tuple[str, ...],
) -> str:
    """Check that a dtype name is allowed; name the operation in any error.

    ``parameter`` adds argument-specific context when supplied. Return the
    accepted name unchanged so callers can use it after validation.

    Parameters
    ----------
    dtype_name : str
        Canonical type name already normalized by the backend.
    operation : str
        Common operation name used to identify the failing call.
    parameter : str or None
        Public argument name for a parameter-specific diagnostic;
        ``None`` describes the operation's dtype support generally.
    supported_dtype_names : tuple of str
        Canonical names accepted by this particular common operation
        or operand.
    """

    if dtype_name not in supported_dtype_names:
        supported = ", ".join(supported_dtype_names)
        subject = "dtypes" if parameter is None else f"{parameter} dtypes"
        raise TypeError(
            f"cuda.coop.{operation} supports {subject} {supported} through the "
            "common API; "
            f"use a backend-qualified import for backend-specific {subject}"
        )
    return dtype_name


def validate_common_numeric_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str | None = None,
) -> str:
    """Require one of the common API's integer or floating-point type names.

    Backend adapters call this after translating compiler-specific type
    objects to names. Checking the same names in one place keeps the
    common API's numeric contract consistent even when backends have
    different type representations.

    Callers must normalize aliases before calling this function.
    Backend-qualified APIs may support additional types outside this
    common set.

    Parameters
    ----------
    dtype_name : str
        Canonical numeric type name, such as ``int32`` or
        ``float64``; aliases must already be resolved.
    operation : str
        Common operation name included in any rejection.
    parameter : str or None, optional
        Operand name included in the diagnostic when only one
        argument's type is being checked.
    """

    return _validate_common_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
        supported_dtype_names=_COMMON_NUMERIC_DTYPE_NAMES,
    )


def validate_common_integer_value_dtype_name(
    dtype_name: str,
    *,
    operation: str,
    parameter: str = "value",
) -> str:
    """Validate a normalized dtype name for a common integer control.

    Counts and offsets accept signed or unsigned 8-, 16-, 32- and 64-bit
    integers. Value ranges depend on the operation and are checked by its
    caller; this helper returns the dtype name unchanged or reports an
    unsupported kind.
    """

    return _validate_common_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
        supported_dtype_names=_COMMON_INTEGER_VALUE_DTYPE_NAMES,
    )

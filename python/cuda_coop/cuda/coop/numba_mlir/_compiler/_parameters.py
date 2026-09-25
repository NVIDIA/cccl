# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trace-static parameter normalization for Numba-CUDA-MLIR lowering.

This module canonicalizes dimensions, dtypes, portable dtype profiles, and
typed scalar literals before provider construction.  It does not inspect IR,
infer launch metadata, or own persistent cache formats.
"""

import math
import operator
from collections import namedtuple
from numbers import Real
from typing import Union

import numpy as np
from numba_cuda_mlir import types as numba_mlir_types

from cuda.coop._core.dtype_policy import (
    validate_portable_numeric_dtype_name,
)

dim3 = namedtuple("dim3", ("x", "y", "z"))


def normalize_dim_param(dim) -> dim3:
    """Normalize a positive one-, two-, or three-dimensional extent."""

    if isinstance(dim, dim3):
        values = tuple(dim)
    elif isinstance(dim, tuple):
        if not 1 <= len(dim) <= 3:
            raise ValueError(
                f"Tuple dimension must have one, two, or three elements; got {len(dim)}"
            )
        values = dim
    else:
        values = (dim,)

    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("Dimension values must be integers")
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise TypeError("Dimension values must be integers") from exc
        if value <= 0:
            raise ValueError(f"Dimension values must be positive, got {dim!r}")
        normalized.append(value)

    normalized.extend([1] * (3 - len(normalized)))
    return dim3(*normalized)


_NP_DTYPE_TO_NUMBA_MLIR_TYPE = {
    np.dtype(np.bool_): numba_mlir_types.boolean,
    np.dtype(np.int8): numba_mlir_types.int8,
    np.dtype(np.int16): numba_mlir_types.int16,
    np.dtype(np.int32): numba_mlir_types.int32,
    np.dtype(np.int64): numba_mlir_types.int64,
    np.dtype(np.uint8): numba_mlir_types.uint8,
    np.dtype(np.uint16): numba_mlir_types.uint16,
    np.dtype(np.uint32): numba_mlir_types.uint32,
    np.dtype(np.uint64): numba_mlir_types.uint64,
    np.dtype(np.float16): numba_mlir_types.float16,
    np.dtype(np.float32): numba_mlir_types.float32,
    np.dtype(np.float64): numba_mlir_types.float64,
    np.dtype(np.complex64): numba_mlir_types.complex64,
    np.dtype(np.complex128): numba_mlir_types.complex128,
}

_NUMBA_MLIR_TYPE_NAME_ALIASES = {
    "bool_": "boolean",
    "bool": "boolean",
}


def _normalize_numba_mlir_type_name(type_name: str) -> str:
    return _NUMBA_MLIR_TYPE_NAME_ALIASES.get(type_name, type_name)


def _dtype_from_numpy(np_dtype: np.dtype) -> numba_mlir_types.Type:
    canonical = np.dtype(np_dtype)
    if canonical in _NP_DTYPE_TO_NUMBA_MLIR_TYPE:
        return _NP_DTYPE_TO_NUMBA_MLIR_TYPE[canonical]

    type_name = _normalize_numba_mlir_type_name(canonical.name)
    if hasattr(numba_mlir_types, type_name):
        resolved = getattr(numba_mlir_types, type_name)
        if isinstance(resolved, numba_mlir_types.Type):
            return resolved

    raise ValueError(f"Unsupported numpy dtype: {canonical}")


def normalize_dtype_param(
    dtype: Union[str, type, "np.dtype", "numba_mlir_types.Type"],
) -> "numba_mlir_types.Type":
    """Normalize a dtype parameter into a Numba-CUDA-MLIR type object."""

    if dtype is bool:
        return numba_mlir_types.boolean
    if dtype is int:
        return numba_mlir_types.int32
    if dtype is float:
        return numba_mlir_types.float32
    if dtype is complex:
        return numba_mlir_types.complex128
    if isinstance(dtype, numba_mlir_types.Type):
        return dtype
    if isinstance(dtype, np.dtype):
        return _dtype_from_numpy(dtype)
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return _dtype_from_numpy(np.dtype(dtype))
    if isinstance(dtype, str):
        if dtype.startswith("np."):
            np_type_name = dtype[3:]
            if not hasattr(np, np_type_name):
                raise ValueError(f"Invalid numpy dtype: {np_type_name}")
            return _dtype_from_numpy(np.dtype(getattr(np, np_type_name)))

        for prefix in ("numba_cuda_mlir.types.", "types."):
            if dtype.startswith(prefix):
                dtype = dtype[len(prefix) :]
                break

        type_name = _normalize_numba_mlir_type_name(dtype)
        if hasattr(numba_mlir_types, type_name):
            resolved = getattr(numba_mlir_types, type_name)
            if isinstance(resolved, numba_mlir_types.Type):
                return resolved
        raise ValueError(f"Invalid Numba-CUDA-MLIR type name: {dtype}")

    raise ValueError(f"Unrecognized dtype format: {dtype}")


_NUMBA_MLIR_DTYPE_NAMES = {
    numba_mlir_type: np_dtype.name
    for np_dtype, numba_mlir_type in _NP_DTYPE_TO_NUMBA_MLIR_TYPE.items()
}


def _normalize_common_dtype(dtype):
    """Return a backend dtype and its portable normalized name."""

    dtype = normalize_dtype_param(dtype)
    return dtype, _NUMBA_MLIR_DTYPE_NAMES.get(dtype, str(dtype))


def _validate_common_numeric_dtype(
    dtype,
    *,
    operation: str,
    parameter: str | None = None,
):
    """Return one normalized dtype from the portable numeric profile."""

    dtype, dtype_name = _normalize_common_dtype(dtype)
    validate_portable_numeric_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
    )
    return dtype


def _python_scalar_dtype(value):
    """Return the compiler dtype of an ordinary or NumPy scalar."""

    if type(value) not in {bool, int, float, complex} and not isinstance(
        value, np.generic
    ):
        return None
    try:
        return normalize_dtype_param(np.asarray(value).dtype)
    except (TypeError, ValueError):
        return None


def _scalar_cast_dtype(function):
    """Return the dtype named by a scalar cast callable, if any."""

    if isinstance(function, numba_mlir_types.Type):
        try:
            return normalize_dtype_param(function)
        except (TypeError, ValueError):
            return None
    try:
        np_dtype = np.dtype(function)
    except (TypeError, ValueError):
        return None
    if np_dtype.subdtype is not None or np_dtype.fields is not None:
        return None
    try:
        return normalize_dtype_param(np_dtype)
    except (TypeError, ValueError):
        return None


def _scalar_operator_result_dtype(function, *operand_dtypes):
    """Ask the active Numba typing context for an expression result dtype."""

    if (
        function is None
        or not operand_dtypes
        or any(dtype is None for dtype in operand_dtypes)
    ):
        return None
    try:
        normalized = tuple(normalize_dtype_param(dtype) for dtype in operand_dtypes)
        from numba_cuda_mlir.descriptor import mlir_target

        mlir_target.ensure_initialized()
        signature = mlir_target.typing_context.resolve_function_type(
            function,
            normalized,
            {},
        )
        if signature is None:
            return None
        return normalize_dtype_param(signature.return_type)
    except Exception:
        # This is best-effort provenance, not the authoritative typing pass.
        return None


def _validate_runtime_integer_dtype(dtype, *, operation: str, parameter: str):
    """Validate the runtime integer domain accepted by Load/Store controls."""

    if isinstance(dtype, numba_mlir_types.Literal):
        dtype = dtype.literal_type
    if isinstance(dtype, numba_mlir_types.Boolean) or not isinstance(
        dtype, numba_mlir_types.Integer
    ):
        raise TypeError(
            f"coop {operation} {parameter} must be an integer, not bool "
            "or a noninteger scalar"
        )
    if dtype.bitwidth > 64 or (not dtype.signed and dtype.bitwidth > 32):
        raise TypeError(
            f"coop {operation} {parameter} must be a signed integer up to "
            "64 bits or an unsigned integer up to 32 bits"
        )
    return dtype


def coerce_static_scalar(
    value: object,
    dtype,
    *,
    operation: str,
    parameter: str,
    source_dtype=None,
):
    """Validate and normalize one trace-static scalar for a target dtype."""

    target_dtype = _validate_common_numeric_dtype(
        dtype,
        operation=operation,
        parameter=parameter,
    )
    target_numpy_dtype = np.dtype(_NUMBA_MLIR_DTYPE_NAMES[target_dtype])

    if source_dtype is None and isinstance(value, np.generic):
        source_dtype = value.dtype
    if source_dtype is not None:
        try:
            normalized_source = normalize_dtype_param(source_dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"cuda.coop.{operation} {parameter} must be a numeric scalar"
            ) from exc
        if normalized_source != target_dtype:
            raise TypeError(
                f"cuda.coop.{operation} {parameter} dtype "
                f"{normalized_source} does not match payload dtype {target_dtype}"
            )
        scalar = value.item() if isinstance(value, np.generic) else value
        if isinstance(scalar, Real) and not math.isfinite(float(scalar)):
            raise ValueError(f"cuda.coop.{operation} {parameter} must be finite")
        return target_numpy_dtype.type(value)

    if isinstance(value, bool) or type(value) is bool:
        raise TypeError(f"cuda.coop.{operation} {parameter} must not be bool")
    if type(value) not in {int, float}:
        raise TypeError(
            f"cuda.coop.{operation} {parameter} must be an ordinary Python "
            "numeric literal or an exactly typed NumPy/compiler scalar"
        )

    if np.issubdtype(target_numpy_dtype, np.integer):
        if type(value) is float:
            raise TypeError(
                f"cuda.coop.{operation} {parameter} does not permit "
                "float-to-integer conversion"
            )
        bounds = np.iinfo(target_numpy_dtype)
        if not bounds.min <= value <= bounds.max:
            raise ValueError(
                f"cuda.coop.{operation} {parameter} value {value} is outside "
                f"the range of {target_numpy_dtype.name}"
            )
        return target_numpy_dtype.type(value)

    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"cuda.coop.{operation} {parameter} must be finite")
    maximum = float(np.finfo(target_numpy_dtype).max)
    if not -maximum <= value <= maximum:
        raise ValueError(
            f"cuda.coop.{operation} {parameter} value {value} is outside "
            f"the finite range of {target_numpy_dtype.name}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        result = target_numpy_dtype.type(value)
    if not np.isfinite(result):
        raise ValueError(
            f"cuda.coop.{operation} {parameter} value {value} is outside "
            f"the finite range of {target_numpy_dtype.name}"
        )
    return result


def _validate_static_oob_default(value: object, dtype):
    """Normalize one compile-time Load default before provider construction."""

    return coerce_static_scalar(
        value,
        dtype,
        operation="load",
        parameter="oob_default",
    )


def _scalar_cpp_literal(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "NAN"
        if np.isposinf(value):
            return "INFINITY"
        if np.isneginf(value):
            return "-INFINITY"
        return repr(value)
    raise ValueError(
        f"Unsupported scalar literal type for compile-time binding: {type(value)}"
    )


def make_typed_cpp_literal(value, dtype):
    """Render a compiler scalar as a C++ literal of exactly ``dtype``."""

    dtype = normalize_dtype_param(dtype)
    from .._types import numba_type_to_cpp

    cpp_type = numba_type_to_cpp(dtype)
    if cpp_type == "storage_t":
        raise ValueError(
            "Compile-time scalar literal binding does not support user-defined dtypes"
        )
    return f"static_cast<{cpp_type}>({_scalar_cpp_literal(value)})"

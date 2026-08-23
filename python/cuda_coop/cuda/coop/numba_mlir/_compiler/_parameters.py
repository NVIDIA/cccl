# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trace-static parameter normalization for Numba-CUDA-MLIR lowering.

This module canonicalizes dimensions, dtypes, portable dtype profiles, and
typed scalar literals before provider construction.  It does not inspect IR,
infer launch metadata, or own persistent cache formats.
"""

import operator
from collections import namedtuple
from typing import Union

import numpy as np
from numba_cuda_mlir import types as numba_mlir_types

from cuda.coop._core.dtype_policy import (
    validate_portable_integer_key_dtype_name,
    validate_portable_numeric_dtype_name,
)

dim3 = namedtuple("dim3", ("x", "y", "z"))

CUB_BLOCK_REDUCE_ALGOS = {
    "raking_commutative_only": (
        "::cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY"
    ),
    "raking": "::cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING",
    "warp_reductions": ("::cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS"),
}

CUB_BLOCK_SCAN_ALGOS = {
    "raking": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING",
    "raking_memoize": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE",
    "warp_scans": "::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS",
}


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


def _validate_common_integer_key_dtype(
    dtype,
    *,
    operation: str,
    parameter: str = "keys",
):
    """Return one normalized dtype from the portable integer-key profile."""

    dtype, dtype_name = _normalize_common_dtype(dtype)
    validate_portable_integer_key_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
    )
    return dtype


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

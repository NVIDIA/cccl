# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dtype normalization for Numba-CUDA-MLIR lowering.

This module canonicalizes public dtype spellings before compiler lowering.
"""

from typing import Union

import numpy as np
from numba_cuda_mlir import types as numba_mlir_types

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

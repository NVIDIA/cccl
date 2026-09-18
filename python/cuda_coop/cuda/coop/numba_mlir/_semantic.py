# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-local semantic normalization for Numba-CUDA-MLIR values."""

from numba_cuda_mlir import types
from numba_cuda_mlir.descriptor import MLIRDispatcher

from cuda.coop._core import semantic_token


def _normalize_numba_callable(value):
    if isinstance(value, MLIRDispatcher):
        return value.py_func
    return value


def _numba_semantic_token(value):
    if isinstance(value, types.Type):
        value = (
            "numba-cuda-mlir-type",
            type(value).__module__,
            type(value).__qualname__,
            str(value),
        )
    else:
        value = _normalize_numba_callable(value)
    return semantic_token(value)

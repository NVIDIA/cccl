# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalize compiler dtypes for cache keys."""

from numba_cuda_mlir import types

from cuda.coop._core import semantic_token


def _numba_semantic_token(value):
    if isinstance(value, types.Type):
        value = (
            "numba-cuda-mlir-type",
            type(value).__module__,
            type(value).__qualname__,
            str(value),
        )
    return semantic_token(value)

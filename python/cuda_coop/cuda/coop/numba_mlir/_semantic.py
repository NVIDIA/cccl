# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalize compiler dtypes for cache keys."""

import numba_cuda_mlir.numba_cuda.types as numba_types

from cuda.coop._core import semantic_token


def _numba_semantic_token(value):
    if isinstance(value, numba_types.Type):
        value = (
            "numba-cuda-mlir-type",
            type(value).__module__,
            type(value).__qualname__,
            str(value),
        )
    return semantic_token(value)

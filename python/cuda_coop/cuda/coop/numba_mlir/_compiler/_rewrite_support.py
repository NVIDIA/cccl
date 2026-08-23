# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Foundation support for cooperative storage constructor rewriting."""

from __future__ import annotations

import operator
import struct
from dataclasses import dataclass
from itertools import count

from numba_cuda_mlir.numba_cuda.core import errors as _numba_errors

_INFERENCE_EXCEPTIONS = (
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    _numba_errors.ConstantInferenceError,
)
_GLOBAL_NAME_COUNTER = count()
_MIN_TEMP_STORAGE_ALIGNMENT = max(1, struct.calcsize("P"))


class CoopSinglePhaseRewriteError(Exception):
    """Raised when a cooperative storage call cannot be rewritten."""


def _next_global_name(stem: str) -> str:
    return f"__cuda_coop_numba_mlir_{stem}_{next(_GLOBAL_NAME_COUNTER)}__"


def _normalize_alignment(
    value: object,
    *,
    context: str,
    minimum: int = 1,
    promote_to_minimum: bool = True,
) -> int:
    if isinstance(value, bool):
        raise CoopSinglePhaseRewriteError(f"{context} must be an integer or None.")
    try:
        alignment = operator.index(value)
    except TypeError as exc:
        raise CoopSinglePhaseRewriteError(
            f"{context} must be an integer or None."
        ) from exc
    if alignment <= 0:
        raise CoopSinglePhaseRewriteError(f"{context} must be a positive integer.")
    if alignment & (alignment - 1):
        raise CoopSinglePhaseRewriteError(f"{context} must be a power of 2.")
    if alignment % minimum:
        if promote_to_minimum:
            return minimum
        raise CoopSinglePhaseRewriteError(f"{context} must be a multiple of {minimum}.")
    return alignment


@dataclass(frozen=True)
class _ThreadDataSpec:
    items_per_thread: int
    dtype: object
    alignment: int


@dataclass(frozen=True)
class _TempStorageCtorSpec:
    size_in_bytes: int | None
    alignment: int

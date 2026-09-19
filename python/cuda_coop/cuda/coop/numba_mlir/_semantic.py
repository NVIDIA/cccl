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


def _normalize_numba_semantic_value(value):
    if isinstance(value, MLIRDispatcher):
        # Nested callees retain their own compile options, unlike the outer
        # callback that cuda.coop recompiles from its Python function. Inspect
        # only these inputs, not dispatcher caches, locks, or compiler state.
        signatures = None
        if not value._can_compile:
            signatures = tuple(
                (signature.return_type, signature.args)
                for signature in value.nopython_signatures
            )
        return (
            "numba-cuda-mlir-device-callee-v1",
            value.py_func,
            value.targetoptions,
            value.locals,
            signatures,
        )
    if isinstance(value, types.Type):
        # The display name is not unique; key defines Numba type equality.
        return (
            "numba-cuda-mlir-type",
            type(value).__module__,
            type(value).__qualname__,
            value.key,
        )
    return value


def _numba_semantic_token(value):
    # StatefulFunction contains a Numba dtype. Normalize its fields here so
    # compilation-time mutations of Numba's dtype singleton cannot perturb a
    # provider cache key. ``name`` is only a diagnostic label.
    value_type = type(value)
    if (
        value_type.__module__ == f"{__package__}._stateful_function"
        and value_type.__qualname__ == "StatefulFunction"
    ):
        from ._stateful_function import StatefulFunction

        if isinstance(value, StatefulFunction):
            op = value.op.__call__ if isinstance(value.op, type) else value.op
            return (
                "numba-cuda-mlir-stateful-function-v1",
                _numba_semantic_token(op),
                _numba_semantic_token(value.dtype),
            )
    value = _normalize_numba_callable(value)
    return semantic_token(value, normalize=_normalize_numba_semantic_value)

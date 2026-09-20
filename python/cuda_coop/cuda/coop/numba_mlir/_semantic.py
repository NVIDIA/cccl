# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Give compiler values stable identities for specialization and symbol reuse.

Equivalent compiler types should select the same compiled operation without
including their internal bookkeeping. Normalize them to their concrete class
and equality key. Unwrap an outer device callback to its Python function;
nested device dispatchers retain their compile options, locals, and any fixed
signatures because those inputs affect the called implementation. The shared
semantic encoder then produces hashable descriptions for cache keys and symbol
hashes, rather than serializing compiler objects for later reconstruction.
"""

from collections.abc import Hashable

import numba_cuda_mlir.numba_cuda.types as numba_types
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
    if isinstance(value, numba_types.Type):
        # The display name is not unique; key defines Numba type equality.
        return (
            "numba-cuda-mlir-type",
            type(value).__module__,
            type(value).__qualname__,
            value.key,
        )
    return value


def _numba_semantic_token(value: object) -> Hashable:
    """Describe compiler values without serializing compiler bookkeeping.

    Unwrap the outer device dispatcher to its Python function, then encode
    nested values with compiler-specific normalization. Numba types contribute
    their concrete class and equality key. Nested device dispatchers retain
    their function, compile options, locals, and any fixed signatures.

    Parameters
    ----------
    value : object
        Compiler dtype, callback, or other semantic key component.

    Returns
    -------
    Hashable
        Semantic description suitable for compiler cache and symbol
        keys. This is a description, not a digest or a reversible encoding.

    Raises
    ------
    TypeError
        The common encoder encounters an unsupported callable value.
    """

    value = _normalize_numba_callable(value)
    return semantic_token(value, normalize=_normalize_numba_semantic_value)

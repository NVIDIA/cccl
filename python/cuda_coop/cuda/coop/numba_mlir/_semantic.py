# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-local semantic normalization for Numba-CUDA-MLIR values."""

import hashlib

import numpy as np
from numba_cuda_mlir import types
from numba_cuda_mlir.descriptor import MLIRDispatcher

from cuda.coop._core import semantic_token


def _normalize_numba_callable(value):
    if isinstance(value, MLIRDispatcher):
        return value.py_func
    return value


def _numpy_dtype_identity(dtype):
    # dtype.str alone loses record fields and subarray element types. Avoid
    # dtype.descr, which cannot describe overlapping or out-of-order fields.
    fields = None
    if dtype.fields is not None:
        fields = tuple(
            (name, _numpy_dtype_identity(field[0]), *field[1:])
            for name in dtype.names
            for field in (dtype.fields[name],)
        )
    subdtype = dtype.subdtype
    return (
        dtype.str,
        dtype.isalignedstruct,
        fields,
        None if subdtype is None else (_numpy_dtype_identity(subdtype[0]), subdtype[1]),
    )


def _normalize_numba_semantic_value(value):
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError(
                "cuda.coop.numba_mlir callback constants cannot contain "
                "NumPy scalars with object dtypes"
            )
        # Scalar repr loses observable bits, such as a NaN's sign and payload.
        return (
            "numba-cuda-mlir-numpy-scalar-v1",
            _numpy_dtype_identity(value.dtype),
            value.tobytes(),
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(
                "cuda.coop.numba_mlir callback constants cannot contain "
                "NumPy arrays with object dtypes"
            )
        # repr(array) truncates contents and depends on global print options.
        # Hash every logical element, including noncontiguous/reversed views.
        return (
            "numba-cuda-mlir-numpy-array-v1",
            _numpy_dtype_identity(value.dtype),
            value.shape,
            value.strides,
            value.flags.writeable,
            hashlib.sha256(value.tobytes(order="C")).digest(),
        )
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
    value = _normalize_numba_callable(value)
    return semantic_token(value, normalize=_normalize_numba_semantic_value)

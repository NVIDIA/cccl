# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
JIT-compiled operator using Numba.

This module contains:
- _JitOp: Wrapper for JIT-compiled callables
- to_stateless_cccl_op: Compile a Python callable to a CCCL Op
"""

from typing import TYPE_CHECKING, Callable, Hashable

from .._bindings import Op, TypeEnum
from ..op import OpKind
from ..types import _TypeDescriptor
from ._caching import CachableFunction

if TYPE_CHECKING:
    from numba.core.typing import Signature


def _to_numba_type(t):
    """
    Convert a type (struct class, _TypeDescriptor, or Numba type) to a Numba type.

    - Struct classes (with _get_numba_type): triggers lazy Numba registration
    - _TypeDescriptor: converts via numpy dtype
    - Numba types: returned as-is
    """
    # Already a Numba type - return directly
    from numba import types as numba_types

    if isinstance(t, numba_types.Type):
        return t

    # Struct class with lazy Numba registration
    if isinstance(t, type) and hasattr(t, "_get_numba_type"):
        return t._get_numba_type()

    # _TypeDescriptor - convert via numpy dtype
    if isinstance(t, _TypeDescriptor):
        import numpy as np
        from numba.np.numpy_support import from_dtype

        _enum_to_numpy = {
            TypeEnum.INT8: np.dtype("int8"),
            TypeEnum.INT16: np.dtype("int16"),
            TypeEnum.INT32: np.dtype("int32"),
            TypeEnum.INT64: np.dtype("int64"),
            TypeEnum.UINT8: np.dtype("uint8"),
            TypeEnum.UINT16: np.dtype("uint16"),
            TypeEnum.UINT32: np.dtype("uint32"),
            TypeEnum.UINT64: np.dtype("uint64"),
            TypeEnum.FLOAT16: np.dtype("float16"),
            TypeEnum.FLOAT32: np.dtype("float32"),
            TypeEnum.FLOAT64: np.dtype("float64"),
        }
        np_dtype = _enum_to_numpy.get(t._type_enum)
        if np_dtype is not None:
            return from_dtype(np_dtype)

        # Handle complex types (stored with STORAGE enum but name like "complex64", "complex128")
        if t._name in ("complex64", "complex128"):
            return from_dtype(np.dtype(t._name))

        raise ValueError(
            f"Cannot convert TypeDescriptor {t} to Numba type. "
            "For struct types, use gpu_struct() instead of custom_type()."
        )

    # Last resort - assume Numba can handle it
    from numba.core.extending import as_numba_type

    return as_numba_type(t)


class _JitOp:
    """
    Internal wrapper for JIT-compiled callables using Numba.

    This wraps Python callables that will be compiled to LTOIR via Numba
    when the compile() method is called. Numba imports are deferred to
    compile time to allow cuda.compute to be imported without Numba.

    Note: This class deliberately does NOT inherit from _BaseOp to avoid
    circular imports. Instead, make_op_adapter() checks for _JitOp explicitly.
    """

    __slots__ = ["_func", "_cachable"]

    def __init__(self, func: Callable):
        self._func = func
        self._cachable = CachableFunction(func)

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        return self._cachable

    def compile(self, input_types, output_type=None) -> Op:
        """Compile this operator using Numba JIT."""
        from .numba_utils import get_inferred_return_type, signature_from_annotations

        # Convert types to Numba types (handles struct classes and TypeDescriptors)
        numba_input_types = tuple(_to_numba_type(t) for t in input_types)
        numba_output_type = (
            _to_numba_type(output_type) if output_type is not None else None
        )

        # Try to get signature from annotations first
        try:
            sig = signature_from_annotations(self._func)
        except ValueError:
            # Infer signature from input/output types
            if numba_output_type is None or (
                hasattr(numba_output_type, "is_internal")
                and not numba_output_type.is_internal
            ):
                numba_output_type = get_inferred_return_type(
                    self._func, numba_input_types
                )
            sig = numba_output_type(*numba_input_types)

        return to_stateless_cccl_op(self._func, sig)

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func


# Public alias
JitOp = _JitOp


def to_stateless_cccl_op(op, sig: "Signature") -> Op:
    """Compile a Python callable to a CCCL Op using Numba.

    Args:
        op: A Python callable to compile
        sig: Numba signature for the function

    Returns:
        Compiled Op object for C++ interop
    """
    from numba import cuda

    from .odr_helpers import create_op_void_ptr_wrapper

    wrapped_op, wrapper_sig = create_op_void_ptr_wrapper(op, sig)

    ltoir, _ = cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")
    return Op(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
    )

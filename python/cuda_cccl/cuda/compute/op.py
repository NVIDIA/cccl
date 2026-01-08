# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Hashable

from ._bindings import Op, OpKind, TypeEnum
from ._caching import CachableFunction
from .types import _TypeDescriptor


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


class _OpAdapter:
    """
    Provides a unified interface for operators, whether they are:
    - Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
    - Stateless user-provided callables
    """

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        raise NotImplementedError("Subclasses must implement this method")

    def compile(self, input_types, output_type=None) -> Op:
        """
        Compile this operator to an Op for CCCL interop.

        Args:
            input_types: Tuple of numba types for input arguments
            output_type: Optional numba type for return value (inferred if None)

        Returns:
            Compiled Op object for C++ interop
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def func(self) -> Callable | None:
        """The underlying callable, if any."""
        return None


class _WellKnownOp(_OpAdapter):
    """Internal wrapper for well-known OpKind values."""

    __slots__ = ["_kind"]

    def __init__(self, kind: OpKind):
        if not _is_well_known_op(kind):
            raise ValueError(
                f"OpKind.{kind.name} is not a well-known operation. "
                "Use OpKind.PLUS, OpKind.MAXIMUM, etc."
            )
        self._kind = kind

    def get_cache_key(self) -> Hashable:
        return (self._kind.name, self._kind.value)

    def compile(self, input_types, output_type=None) -> Op:
        return Op(
            operator_type=self._kind,
            name="",
            ltoir=b"",
            state_alignment=1,
            state=b"",
        )

    @property
    def kind(self) -> OpKind:
        """The underlying OpKind."""
        return self._kind


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


class _JitOp(_OpAdapter):
    """
    Internal wrapper for JIT-compiled callables using Numba.

    This wraps Python callables that will be compiled to LTOIR via Numba
    when the compile() method is called. Numba imports are deferred to
    compile time to allow cuda.compute to be imported without Numba.
    """

    __slots__ = ["_func", "_cachable"]

    def __init__(self, func: Callable):
        self._func = func
        self._cachable = CachableFunction(func)

    def get_cache_key(self) -> Hashable:
        return self._cachable

    def compile(self, input_types, output_type=None) -> Op:
        # Lazy import of Numba-dependent code
        from . import _cccl_interop as cccl
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

        return cccl.to_stateless_cccl_op(self._func, sig)

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func


class _CompiledOp(_OpAdapter):
    """
    Pre-compiled operator from LTOIR bytecode.

    This allows users to bring their own compiler (BYOC) by providing
    pre-compiled LTOIR rather than relying on Numba for JIT compilation.

    The LTOIR must follow the CCCL ABI convention where all arguments and
    the return value are passed as void pointers:
        extern "C" __device__ void name(void* arg0, void* arg1, ..., void* result)

    Example:
        from cuda.compute import CompiledOp, types

        # Compile C++ to LTOIR using cuda.core
        ltoir = compile_cpp_to_ltoir(Program, cpp_source, arch)

        add_op = CompiledOp(
            ltoir=ltoir,
            name="my_add",
            arg_types=(types.int32, types.int32),
            return_type=types.int32,
        )

        reduce_into(d_in, d_out, add_op, num_items, h_init)
    """

    __slots__ = ["_ltoir", "_name", "_arg_types", "_return_type"]

    def __init__(
        self,
        ltoir: bytes,
        name: str,
        arg_types: tuple[_TypeDescriptor, ...],
        return_type: _TypeDescriptor,
    ):
        """
        Create a pre-compiled operator from LTOIR bytecode.

        Args:
            ltoir: LTOIR bytecode compiled from C++ source
            name: The symbol name of the device function (must match extern "C" name)
            arg_types: Tuple of type descriptors for the input arguments
            return_type: Type descriptor for the return value
        """
        if not isinstance(ltoir, bytes):
            raise TypeError(f"ltoir must be bytes, got {type(ltoir).__name__}")
        if not ltoir:
            raise ValueError("ltoir cannot be empty")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        if not name:
            raise ValueError("name cannot be empty")
        if not isinstance(arg_types, tuple):
            raise TypeError(
                f"arg_types must be a tuple, got {type(arg_types).__name__}"
            )
        for i, arg_type in enumerate(arg_types):
            if not isinstance(arg_type, _TypeDescriptor):
                raise TypeError(
                    f"arg_types[{i}] must be a TypeDescriptor (e.g., types.int32), "
                    f"got {type(arg_type).__name__}"
                )
        if not isinstance(return_type, _TypeDescriptor):
            raise TypeError(
                f"return_type must be a TypeDescriptor (e.g., types.int32), "
                f"got {type(return_type).__name__}"
            )

        self._ltoir = ltoir
        self._name = name
        self._arg_types = arg_types
        self._return_type = return_type

    def get_cache_key(self) -> Hashable:
        # Use the LTOIR bytes hash and name as cache key
        return (hash(self._ltoir), self._name, self._arg_types, self._return_type)

    def compile(self, input_types, output_type=None) -> Op:
        # Already compiled - just return the Op with the LTOIR
        return Op(
            operator_type=OpKind.STATELESS,
            name=self._name,
            ltoir=self._ltoir,
            state_alignment=1,
            state=b"",
        )

    @property
    def name(self) -> str:
        """The symbol name of the compiled function."""
        return self._name

    @property
    def ltoir(self) -> bytes:
        """The LTOIR bytecode."""
        return self._ltoir

    @property
    def arg_types(self) -> tuple[_TypeDescriptor, ...]:
        """The argument type descriptors."""
        return self._arg_types

    @property
    def return_type(self) -> _TypeDescriptor:
        """The return type descriptor."""
        return self._return_type


# Public aliases
OpAdapter = _OpAdapter
CompiledOp = _CompiledOp


def make_op_adapter(op) -> OpAdapter:
    """
    Create an Op from a callable or well-known OpKind.

    Args:
        op: Callable or OpKind

    Returns:
        A value with appropriate subtype of _BaseOp
    """
    # Already an _OpAdapter instance:
    if isinstance(op, _OpAdapter):
        return op

    # Well-known operation
    if isinstance(op, OpKind):
        return _WellKnownOp(op)

    return _JitOp(op)


__all__ = [
    "CompiledOp",
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
]

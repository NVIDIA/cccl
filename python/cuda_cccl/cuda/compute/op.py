# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

from ._bindings import Op, OpKind
from ._caching import CachableFunction, cache_with_registered_key_functions


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


class _OpAdapter:
    """
    Provides a unified interface for operators, whether they are:
    - Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
    - Stateless user-provided callables
    """

    def compile(self, input_types, output_type=None) -> Op:
        """
        Compile this operator to an Op for CCCL interop.

        Args:
            input_types: Tuple of TypeDescriptors for input arguments
            output_type: Optional TypeDescriptor for return value (inferred if None)

        Returns:
            Compiled Op object for C++ interop
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def func(self) -> Callable | None:
        """The underlying callable, if any."""
        return None

    def get_return_type(self, input_types):
        raise NotImplementedError(
            f"get_return_type not implemented for {self.__class__.__name__}"
        )


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

    def __eq__(self, other):
        if not isinstance(other, _WellKnownOp):
            return False
        return self._kind == other._kind

    def __hash__(self):
        return hash(self._kind)


class _StatelessOp(_OpAdapter):
    """Internal wrapper for stateless callables."""

    __slots__ = ["_func", "_cachable"]

    def __init__(self, func: Callable):
        self._func = func
        self._cachable = CachableFunction(func)

    def get_return_type(self, input_types):
        from ._jit import get_or_infer_return_type

        return get_or_infer_return_type(self._func, input_types)

    def compile(self, input_types, output_type=None) -> Op:
        from ._jit import compile_op

        return compile_op(self._func, input_types, output_type)

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func

    def __eq__(self, other):
        if not isinstance(other, _StatelessOp):
            return False
        return self._cachable == other._cachable

    def __hash__(self):
        return hash(self._cachable)


# Public aliases
OpAdapter = _OpAdapter


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

    return _StatelessOp(op)


__all__ = [
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
]


cache_with_registered_key_functions.register(
    _WellKnownOp, lambda op: (op._kind.name, op._kind.value)
)

cache_with_registered_key_functions.register(_StatelessOp, lambda op: op._cachable)

cache_with_registered_key_functions.register(
    OpKind, lambda kind: (kind.name, kind.value)
)

cache_with_registered_key_functions.register(
    type(lambda: None), lambda func: CachableFunction(func)
)

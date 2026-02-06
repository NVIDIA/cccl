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
    - Stateful user-provided callables
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
    def is_stateful(self) -> bool:
        """Return True if this op has runtime state."""
        return False

    def update_op_state(self, cccl_op: Op) -> None:
        """
        Update the Op's state bytes.

        Args:
            cccl_op: The compiled CCCL Op to update

        Default implementation is a no-op (for stateless ops).
        """
        pass

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
    from ._jit import to_jit_op_adapter

    # Already an _OpAdapter instance:
    if isinstance(op, _OpAdapter):
        return op

    # Well-known operation
    if isinstance(op, OpKind):
        return _WellKnownOp(op)

    # It's a Python callable
    return to_jit_op_adapter(op)


__all__ = [
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
]


cache_with_registered_key_functions.register(
    _WellKnownOp, lambda op: (op._kind.name, op._kind.value)
)

cache_with_registered_key_functions.register(
    OpKind, lambda kind: (kind.name, kind.value)
)

cache_with_registered_key_functions.register(
    type(lambda: None), lambda func: CachableFunction(func)
)

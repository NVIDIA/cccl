# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Operator adapters for cuda.compute algorithms.

This module provides the unified interface for operators used in algorithms like
reduce_into, scan, etc. Operators can be:

- Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
- JIT-compiled callables (using Numba)
- Pre-compiled LTOIR (BYOC - Bring Your Own Compiler)
"""

from typing import Callable, Hashable, Protocol, runtime_checkable

from ._bindings import Op, OpKind

# Re-export from submodules
from .compiled.op import CompiledOp


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


@runtime_checkable
class OpProtocol(Protocol):
    """
    Protocol defining the interface for operator adapters.

    Provides a unified interface for operators, whether they are:
    - Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
    - JIT-compiled user-provided callables (Numba)
    - Pre-compiled LTOIR (BYOC)
    """

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        ...

    def compile(self, input_types, output_type=None) -> Op:
        """
        Compile this operator to an Op for CCCL interop.

        Args:
            input_types: Tuple of types for input arguments
            output_type: Optional type for return value (inferred if None)

        Returns:
            Compiled Op object for C++ interop
        """
        ...

    @property
    def func(self) -> Callable | None:
        """The underlying callable, if any."""
        ...


class _WellKnownOp:
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
    def func(self) -> Callable | None:
        """The underlying callable, if any."""
        return None

    @property
    def kind(self) -> OpKind:
        """The underlying OpKind."""
        return self._kind


def make_op_adapter(op) -> OpProtocol:
    """
    Create an Op adapter from a callable or well-known OpKind.

    Args:
        op: Callable, OpKind, or existing op adapter

    Returns:
        An object implementing OpProtocol
    """
    # Already implements OpProtocol
    if isinstance(op, OpProtocol):
        return op

    # Well-known operation
    if isinstance(op, OpKind):
        return _WellKnownOp(op)

    # JIT-compiled callable - needs Numba
    try:
        import numba.cuda as _  # noqa: F401
    except ImportError:
        raise ImportError(
            "Using Python callables as operators requires numba-cuda, but it is "
            "not installed or failed to import.\n\n"
        ) from None

    from ._numba.op import _JitOp  # noqa: F401

    return _JitOp(op)


# Public aliases for backwards compatibility
OpAdapter = OpProtocol

__all__ = [
    "CompiledOp",
    "OpAdapter",
    "OpKind",
    "OpProtocol",
    "make_op_adapter",
]

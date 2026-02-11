# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from __future__ import annotations

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

    def get_state(self) -> bytes:
        """
        Return the op's state bytes.
        """
        return b""

    def get_return_type(self, input_types):
        """Get the return type for this op given input types."""
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


class RawOp(_OpAdapter):
    """
    ``RawOp`` can be used to directly pass compiled device code (LTO-IR) implementing custom operators.

    This is useful for users who wish to implement custom operators in C++ or another language,
    or wish to use a different compilation pipeline than the default
    (JIT compilation from Python callables using Numba CUDA).

    Example:
        The example below shows how to compile C++ device code to LTOIR and use it with
        :func:`reduce_into <cuda.compute.algorithms.reduce_into>`:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/raw_op/cpp_stateless.py
            :language: python
            :start-after: # example-begin

    Args:
        name: The ABI name of the operator
        ltoir: bytes object containing the LTO-IR of the compiled operator
        extra_ltoirs: Optional list of additional LTOIR modules that the main LTOIR depends on
        state: Optional bytes representing the operator's state
        state_alignment: Alignment requirement for the state bytes (default: 1)
        extra_ltoirs: Optional list of additional LTO-IRs to include during linking

    Notes:
        - The provided LTO-IR must define a function with the specified name and the correct signature.
        - The function must use untyped pointers for all parameters and return type. The function body
          is responsible for correctly interpreting the pointer arguments based on the expected input and output types.
          For stateless operators, the signature is

             void func(void* arg1, void* arg2, ..., void* result)`

          For stateful operators, the first parameter must be a pointer to the state:

             void func(void* state, void* arg1, void* arg2, ...)
    """

    __slots__ = ["_ltoir", "_name", "_state", "_state_alignment", "_extra_ltoirs"]

    def __init__(
        self,
        *,
        ltoir: bytes,
        name: str,
        state: bytes = b"",
        state_alignment: int = 1,
        extra_ltoirs: list[bytes] | None = None,
    ):
        """
        Create an operator from LTO-IR.

        Args:
            ltoir: Compiled LTOIR bytecode from any language/compiler
            name: Function name in the LTOIR
            state: State bytes for stateful operators (default: b"")
            state_alignment: Memory alignment for state (default: 1)
        """
        self._ltoir = ltoir
        self._name = name
        self._state = state
        self._state_alignment = state_alignment
        self._extra_ltoirs = extra_ltoirs or []

    def compile(self, input_types, output_type=None) -> Op:
        # Determine if stateful based on whether state is provided
        op_kind = OpKind.STATEFUL if self._state else OpKind.STATELESS

        return Op(
            operator_type=op_kind,
            name=self._name,
            ltoir=self._ltoir,
            state=self._state,
            state_alignment=self._state_alignment,
            extra_ltoirs=self._extra_ltoirs,
        )

    def get_state(self) -> bytes:
        """Return the op's state bytes."""
        return self._state

    @property
    def _identity(self):
        return (
            self._ltoir,
            self._name,
            self._state,
            self._state_alignment,
            tuple(self._extra_ltoirs),
        )

    def __eq__(self, other):
        if not isinstance(other, RawOp):
            return False
        return self._identity == other._identity

    def __hash__(self):
        return hash(self._identity)


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


cache_with_registered_key_functions.register(
    _WellKnownOp, lambda op: (op._kind.name, op._kind.value)
)

cache_with_registered_key_functions.register(
    OpKind, lambda kind: (kind.name, kind.value)
)

cache_with_registered_key_functions.register(
    type(lambda: None), lambda func: CachableFunction(func)
)

cache_with_registered_key_functions.register(RawOp, lambda op: (op._identity))


__all__ = [
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
    "RawOp",
]

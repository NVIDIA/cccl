# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from __future__ import annotations

from ._bindings import Op, OpKind
from ._caching import CachableFunction, cache_with_registered_key_functions
from ._device_code import DeviceCode


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
    ``RawOp`` lets you supply pre-compiled device code (LLVM bitcode or LTO-IR)
    that implements a custom operator, bypassing the default Numba-based JIT
    pipeline.

    For ``cuda.compute`` v2, **LLVM bitcode is the preferred form** — it is
    linked into the CUB module at the LLVM IR level, so the optimizer inlines
    the operator into kernel inner loops. LTO-IR is supported as an escape
    hatch for callers with pre-built ``nvcc -dlto`` artifacts; LTO-IR
    operators are routed through ``nvJitLink`` and remain a real ``CALL``
    target in the generated SASS — functionally correct, but materially
    worse code-gen (more registers, more spills, no cross-boundary inlining).

    Example:
        Supplying LLVM bitcode (the recommended v2 path):

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/raw_op/llvm_stateless.py
            :language: python
            :start-after: # example-begin

        Supplying C++ device code compiled to LTO-IR via NVRTC:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/raw_op/cpp_stateless.py
            :language: python
            :start-after: # example-begin

    Args:
        name: The ABI name of the operator.
        ltoir: Either raw ``bytes`` (treated as LTO-IR — the legacy form) or a
            :class:`DeviceCode` wrapping ``(bytes_, kind)`` where ``kind`` is
            one of ``"ltoir"``, ``"llvm_ir"`` (recommended for v2), or
            ``"cpp_source"``. The kwarg is named ``ltoir`` for historical
            reasons; supply a ``DeviceCode`` for anything other than LTO-IR.
        state: Optional bytes representing the operator's state.
        state_alignment: Alignment requirement for the state bytes (default: 1).
        extra_ltoirs: Optional list of extras to link. Each entry is either raw
            ``bytes`` (LTO-IR) or a :class:`DeviceCode`. Forms may be mixed.

    Notes:
        - The provided code must define a function with the specified name and the correct signature.
        - The function must use untyped pointers for all parameters and return type. The function body
          is responsible for correctly interpreting the pointer arguments based on the expected input and output types.
          For stateless operators, the signature is

             void func(void* arg1, void* arg2, ..., void* result)`

          For stateful operators, the first parameter must be a pointer to the state:

             void func(void* state, void* arg1, void* arg2, ...)
    """

    __slots__ = [
        "_ltoir",
        "_name",
        "_state",
        "_state_alignment",
        "_extra_ltoirs",
    ]

    def __init__(
        self,
        *,
        ltoir: bytes | DeviceCode,
        name: str,
        state: bytes = b"",
        state_alignment: int = 1,
        extra_ltoirs: list[bytes | DeviceCode] | None = None,
    ):
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


def _jit_op_adapter_factory():
    # helper that tries to import `_jit.py`. If it fails,
    # returns a function that raises an appropriate error when called.
    try:
        from ._jit import to_jit_op_adapter

        return to_jit_op_adapter
    except ModuleNotFoundError as e:
        if "numba" in str(e):

            def _missing_jit_adapter(op):
                raise ImportError(
                    "numba-cuda is required to JIT compile Python callables"
                )

            return _missing_jit_adapter
        raise


to_jit_op_adapter = _jit_op_adapter_factory()


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

cache_with_registered_key_functions.register(RawOp, lambda op: op._identity)


__all__ = [
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
    "RawOp",
]

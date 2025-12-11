# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Hashable

import numba

from ._bindings import Op, OpKind
from ._caching import CachableFunction
from ._stateful import maybe_transform_to_stateful


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


class _OpAdapter:
    """
    Provides a unified interface for operators, whether they are:
    - Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
    - Stateless user-provided callables
    - Stateful user-provided callables
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
    def is_stateful(self) -> bool:
        return False

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


class _StatelessOp(_OpAdapter):
    """Internal wrapper for stateless callables."""

    __slots__ = ["_func", "_cachable"]

    def __init__(self, func: Callable):
        self._func = func
        self._cachable = CachableFunction(func)

    def get_cache_key(self) -> Hashable:
        return self._cachable

    def compile(self, input_types, output_type=None) -> Op:
        from . import _cccl_interop as cccl
        from .numba_utils import get_inferred_return_type, signature_from_annotations

        # Try to get signature from annotations first
        try:
            sig = signature_from_annotations(self._func)
        except ValueError:
            # Infer signature from input/output types
            if output_type is None or (
                hasattr(output_type, "is_internal") and not output_type.is_internal
            ):
                output_type = get_inferred_return_type(self._func, input_types)
            sig = output_type(*input_types)

        return cccl.to_stateless_cccl_op(self._func, sig)

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func


class _StatefulOp(_OpAdapter):
    """
    Operator with one or more state arrays that can be read and modified
    during algorithm execution.

    Args:
        func: Callable that takes regular arguments followed by state arguments.
            For example, a stateful binary operator would be
            ``func(lhs, rhs, state1, state2, ...) -> result``.
        *state_arrays: One or more device arrays containing state.
            Must be device arrays.
    """

    __slots__ = ["_func", "_state_arrays", "_cachable"]

    def __init__(self, func, *state_arrays):
        self._func = func
        self._state_arrays = state_arrays
        self._cachable = CachableFunction(func)

    def get_cache_key(self) -> Hashable:
        from ._utils import protocols

        state_info = tuple((protocols.get_dtype(s), s.size) for s in self._state_arrays)
        return (self.__class__.__name__, self._cachable, state_info)

    def compile(self, input_types, output_type=None) -> Op:
        from . import _cccl_interop as cccl
        from ._utils import protocols
        from .numba_utils import get_inferred_return_type

        # Infer output type if needed
        if output_type is None or (
            hasattr(output_type, "is_internal") and not output_type.is_internal
        ):
            output_type = get_inferred_return_type(self._func, input_types)

        # Build signature including state arrays
        state_array_types = [
            numba.types.Array(numba.from_dtype(protocols.get_dtype(s)), 1, "A")
            for s in self._state_arrays
        ]
        sig = output_type(*input_types, *state_array_types)
        return cccl.to_stateful_cccl_op(self._func, self._state_arrays, sig)

    @property
    def is_stateful(self) -> bool:
        return True

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func

    @property
    def state(self):
        """Access the device state array(for single-state ops) or tuple of arrays."""
        if len(self._state_arrays) == 1:
            return self._state_arrays[0]
        return self._state_arrays

    def set_state(self, *state_arrays):
        """Set the state arrays."""
        self._state_arrays = state_arrays

    @property
    def state_arrays(self):
        """Access all state arrays as a tuple."""
        return self._state_arrays


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

    transformed_func, state_arrays = maybe_transform_to_stateful(op)
    if state_arrays:
        return _StatefulOp(transformed_func, *state_arrays)
    else:
        # no state arrays, return as stateless op
        return _StatelessOp(transformed_func)


__all__ = [
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
]

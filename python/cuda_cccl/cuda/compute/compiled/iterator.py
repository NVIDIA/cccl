# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
CompiledIterator - Iterator with pre-compiled LTOIR for advance/dereference.

This enables users to bring their own compiler (BYOC) by providing
pre-compiled LTOIR rather than relying on Numba for JIT compilation.
"""

from typing import Union

import numpy as np

from .._bindings import Iterator, IteratorKind, IteratorState, Op, OpKind
from ..types import _TypeDescriptor
from .op import CompiledOp

# Type alias for state input - can be bytes or numpy scalar/array
StateInput = Union[bytes, np.ndarray, np.generic]


def _convert_state(state: StateInput) -> tuple[bytes, int | None]:
    """
    Convert state input to (bytes, alignment) tuple.

    Args:
        state: Raw bytes, numpy scalar, or numpy array

    Returns:
        Tuple of (state_bytes, alignment)
    """
    if isinstance(state, bytes):
        # Raw bytes - caller must provide alignment separately
        return state, None

    if isinstance(state, (np.ndarray, np.generic)):
        # Numpy scalar or array - convert to bytes and infer alignment
        arr = np.asarray(state)
        if arr.ndim > 0 and arr.size != 1:
            # For arrays, ensure contiguous and get bytes
            arr = np.ascontiguousarray(arr)
        state_bytes = arr.tobytes()
        alignment = arr.dtype.alignment
        return state_bytes, alignment

    raise TypeError(
        f"state must be bytes, numpy scalar, or numpy array, got {type(state).__name__}"
    )


class CompiledIterator:
    """
    Iterator with pre-compiled advance and dereference operations.

    This allows users to bring their own compiler (BYOC) by providing
    pre-compiled LTOIR for the advance and dereference methods rather
    than relying on Numba for JIT compilation.

    The LTOIR functions must follow the CCCL ABI conventions:
    - advance: extern "C" __device__ void name(void* state_ptr, void* offset_ptr)
    - dereference (input): extern "C" __device__ void name(void* state_ptr, void* result_ptr)
    - dereference (output): extern "C" __device__ void name(void* state_ptr, void* value_ptr)

    Example:
        from cuda.compute import CompiledIterator, CompiledOp, types
        from cuda.core import Program, ProgramOptions

        # Compile advance/dereference to LTOIR
        advance_op = CompiledOp(advance_ltoir, "advance")
        deref_op = CompiledOp(deref_ltoir, "dereference")

        # Create iterator with numpy state (alignment auto-inferred)
        counting_iter = CompiledIterator(
            state=np.int64(10),
            value_type=types.int64,
            advance=advance_op,
            input_dereference=deref_op,
        )

        reduce_into(counting_iter, d_out, OpKind.PLUS, 5, h_init)
    """

    __slots__ = [
        "_state",
        "_state_alignment",
        "_value_type",
        "_advance",
        "_input_dereference",
        "_output_dereference",
    ]

    def __init__(
        self,
        state: StateInput,
        value_type: _TypeDescriptor,
        advance: CompiledOp,
        input_dereference: CompiledOp | None = None,
        output_dereference: CompiledOp | None = None,
        *,
        state_alignment: int | None = None,
    ):
        """
        Create a pre-compiled iterator from LTOIR bytecode.

        Args:
            state: The iterator state as raw bytes, numpy scalar, or numpy array.
                   When numpy is used, alignment is automatically inferred.
            value_type: Type descriptor for the iterator's value type
            advance: CompiledOp for the advance function
            input_dereference: Optional CompiledOp for input dereference
            output_dereference: Optional CompiledOp for output dereference
            state_alignment: Alignment requirement for the state in bytes.
                            Required when state is raw bytes, optional when numpy.
        """
        # Convert state and get alignment
        state_bytes, inferred_alignment = _convert_state(state)

        # Determine final alignment
        if state_alignment is not None:
            final_alignment = state_alignment
        elif inferred_alignment is not None:
            final_alignment = inferred_alignment
        else:
            raise ValueError("state_alignment is required when state is raw bytes")

        # Validate alignment
        if not isinstance(final_alignment, int) or final_alignment < 1:
            raise ValueError(
                f"state_alignment must be a positive integer, got {final_alignment}"
            )
        if final_alignment & (final_alignment - 1) != 0:
            raise ValueError(
                f"state_alignment must be a power of 2, got {final_alignment}"
            )

        # Validate value_type
        if not isinstance(value_type, _TypeDescriptor):
            raise TypeError(
                f"value_type must be a TypeDescriptor (e.g., types.int64), "
                f"got {type(value_type).__name__}"
            )

        # Validate advance
        if not isinstance(advance, CompiledOp):
            raise TypeError(
                f"advance must be a CompiledOp, got {type(advance).__name__}"
            )

        # Validate input_dereference
        if input_dereference is not None and not isinstance(
            input_dereference, CompiledOp
        ):
            raise TypeError(
                f"input_dereference must be a CompiledOp, "
                f"got {type(input_dereference).__name__}"
            )

        # Validate output_dereference
        if output_dereference is not None and not isinstance(
            output_dereference, CompiledOp
        ):
            raise TypeError(
                f"output_dereference must be a CompiledOp, "
                f"got {type(output_dereference).__name__}"
            )

        # At least one dereference must be provided
        if input_dereference is None and output_dereference is None:
            raise ValueError(
                "At least one of input_dereference or output_dereference must be provided"
            )

        self._state = state_bytes
        self._state_alignment = final_alignment
        self._value_type = value_type
        self._advance = advance
        self._input_dereference = input_dereference
        self._output_dereference = output_dereference

    @property
    def state(self) -> bytes:
        """The iterator state as raw bytes."""
        return self._state

    @property
    def state_alignment(self) -> int:
        """Alignment requirement for the state in bytes."""
        return self._state_alignment

    @property
    def value_type(self) -> _TypeDescriptor:
        """Type descriptor for the iterator's value type."""
        return self._value_type

    @property
    def is_input_iterator(self) -> bool:
        """Whether this iterator can be used as an input iterator."""
        return self._input_dereference is not None

    @property
    def is_output_iterator(self) -> bool:
        """Whether this iterator can be used as an output iterator."""
        return self._output_dereference is not None

    def to_cccl_iter(self, is_output: bool = False) -> Iterator:
        """
        Convert this iterator to a CCCL Iterator object.

        Args:
            is_output: If True, use output_dereference; otherwise use input_dereference

        Returns:
            CCCL Iterator object for C++ interop
        """
        # Create advance Op from CompiledOp
        advance_op = Op(
            operator_type=OpKind.STATELESS,
            name=self._advance.name,
            ltoir=self._advance.ltoir,
        )

        # Create dereference Op based on direction
        if is_output:
            if self._output_dereference is None:
                raise ValueError("This iterator does not support output operations")
            deref_op = Op(
                operator_type=OpKind.STATELESS,
                name=self._output_dereference.name,
                ltoir=self._output_dereference.ltoir,
            )
        else:
            if self._input_dereference is None:
                raise ValueError("This iterator does not support input operations")
            deref_op = Op(
                operator_type=OpKind.STATELESS,
                name=self._input_dereference.name,
                ltoir=self._input_dereference.ltoir,
            )

        # Wrap state bytes in IteratorState
        iter_state = IteratorState(self._state)

        # Create the CCCL Iterator
        return Iterator(
            self._state_alignment,
            IteratorKind.ITERATOR,
            advance_op,
            deref_op,
            self._value_type.to_type_info(),
            state=iter_state,
        )

    @property
    def kind(self):
        """
        Return a hashable kind for caching purposes.

        This is used by algorithms to create cache keys based on iterator type.
        """
        return (
            "CompiledIterator",
            self._advance.get_cache_key(),
            self._input_dereference.get_cache_key()
            if self._input_dereference
            else None,
            self._output_dereference.get_cache_key()
            if self._output_dereference
            else None,
            self._value_type,
        )

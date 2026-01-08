# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
CompiledIterator - Iterator with pre-compiled LTOIR for advance/dereference.

This enables users to bring their own compiler (BYOC) by providing
pre-compiled LTOIR rather than relying on Numba for JIT compilation.
"""

from typing import Callable

from .._bindings import Iterator, IteratorKind, IteratorState, Op, OpKind
from ..types import _TypeDescriptor


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
        from cuda.compute import CompiledIterator, types

        # Compile advance/dereference to LTOIR
        advance_ltoir = compile_cpp_to_ltoir(Program, advance_source, arch)
        deref_ltoir = compile_cpp_to_ltoir(Program, deref_source, arch)

        # Create iterator state (e.g., starting count value)
        state = np.int64(10).tobytes()

        counting_iter = CompiledIterator(
            state=state,
            state_alignment=8,
            value_type=types.int64,
            advance=("counting_advance", advance_ltoir),
            input_dereference=("counting_deref", deref_ltoir),
        )

        reduce_into(counting_iter, d_out, OpKind.PLUS, 5, h_init)
    """

    __slots__ = [
        "_state",
        "_state_alignment",
        "_value_type",
        "_advance_name",
        "_advance_ltoir",
        "_input_deref_name",
        "_input_deref_ltoir",
        "_output_deref_name",
        "_output_deref_ltoir",
        "_host_advance_fn",
    ]

    def __init__(
        self,
        state: bytes,
        state_alignment: int,
        value_type: _TypeDescriptor,
        advance: tuple[str, bytes],
        input_dereference: tuple[str, bytes] | None = None,
        output_dereference: tuple[str, bytes] | None = None,
        host_advance: Callable | None = None,
    ):
        """
        Create a pre-compiled iterator from LTOIR bytecode.

        Args:
            state: The iterator state as raw bytes
            state_alignment: Alignment requirement for the state in bytes
            value_type: Type descriptor for the iterator's value type
            advance: Tuple of (name, ltoir) for the advance function
            input_dereference: Optional tuple of (name, ltoir) for input dereference
            output_dereference: Optional tuple of (name, ltoir) for output dereference
            host_advance: Optional host-side advance function for CPU-side iteration
        """
        # Validate state
        if not isinstance(state, bytes):
            raise TypeError(f"state must be bytes, got {type(state).__name__}")

        # Validate state_alignment
        if not isinstance(state_alignment, int) or state_alignment < 1:
            raise ValueError(
                f"state_alignment must be a positive integer, got {state_alignment}"
            )
        if state_alignment & (state_alignment - 1) != 0:
            raise ValueError(
                f"state_alignment must be a power of 2, got {state_alignment}"
            )

        # Validate value_type
        if not isinstance(value_type, _TypeDescriptor):
            raise TypeError(
                f"value_type must be a TypeDescriptor (e.g., types.int64), "
                f"got {type(value_type).__name__}"
            )

        # Validate advance
        if not isinstance(advance, tuple) or len(advance) != 2:
            raise TypeError("advance must be a tuple of (name, ltoir)")
        advance_name, advance_ltoir = advance
        if not isinstance(advance_name, str) or not advance_name:
            raise ValueError("advance name must be a non-empty string")
        if not isinstance(advance_ltoir, bytes) or not advance_ltoir:
            raise ValueError("advance ltoir must be non-empty bytes")

        # Validate input_dereference
        input_deref_name, input_deref_ltoir = None, None
        if input_dereference is not None:
            if not isinstance(input_dereference, tuple) or len(input_dereference) != 2:
                raise TypeError("input_dereference must be a tuple of (name, ltoir)")
            input_deref_name, input_deref_ltoir = input_dereference
            if not isinstance(input_deref_name, str) or not input_deref_name:
                raise ValueError("input_dereference name must be a non-empty string")
            if not isinstance(input_deref_ltoir, bytes) or not input_deref_ltoir:
                raise ValueError("input_dereference ltoir must be non-empty bytes")

        # Validate output_dereference
        output_deref_name, output_deref_ltoir = None, None
        if output_dereference is not None:
            if (
                not isinstance(output_dereference, tuple)
                or len(output_dereference) != 2
            ):
                raise TypeError("output_dereference must be a tuple of (name, ltoir)")
            output_deref_name, output_deref_ltoir = output_dereference
            if not isinstance(output_deref_name, str) or not output_deref_name:
                raise ValueError("output_dereference name must be a non-empty string")
            if not isinstance(output_deref_ltoir, bytes) or not output_deref_ltoir:
                raise ValueError("output_dereference ltoir must be non-empty bytes")

        # At least one dereference must be provided
        if input_dereference is None and output_dereference is None:
            raise ValueError(
                "At least one of input_dereference or output_dereference must be provided"
            )

        self._state = state
        self._state_alignment = state_alignment
        self._value_type = value_type
        self._advance_name = advance_name
        self._advance_ltoir = advance_ltoir
        self._input_deref_name = input_deref_name
        self._input_deref_ltoir = input_deref_ltoir
        self._output_deref_name = output_deref_name
        self._output_deref_ltoir = output_deref_ltoir
        self._host_advance_fn = host_advance

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
        return self._input_deref_ltoir is not None

    @property
    def is_output_iterator(self) -> bool:
        """Whether this iterator can be used as an output iterator."""
        return self._output_deref_ltoir is not None

    def to_cccl_iter(self, is_output: bool = False) -> Iterator:
        """
        Convert this iterator to a CCCL Iterator object.

        Args:
            is_output: If True, use output_dereference; otherwise use input_dereference

        Returns:
            CCCL Iterator object for C++ interop
        """
        # Create advance Op
        advance_op = Op(
            operator_type=OpKind.STATELESS,
            name=self._advance_name,
            ltoir=self._advance_ltoir,
        )

        # Create dereference Op based on direction
        if is_output:
            if self._output_deref_ltoir is None:
                raise ValueError("This iterator does not support output operations")
            deref_op = Op(
                operator_type=OpKind.STATELESS,
                name=self._output_deref_name,
                ltoir=self._output_deref_ltoir,
            )
        else:
            if self._input_deref_ltoir is None:
                raise ValueError("This iterator does not support input operations")
            deref_op = Op(
                operator_type=OpKind.STATELESS,
                name=self._input_deref_name,
                ltoir=self._input_deref_ltoir,
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
            hash(self._advance_ltoir),
            hash(self._input_deref_ltoir) if self._input_deref_ltoir else None,
            hash(self._output_deref_ltoir) if self._output_deref_ltoir else None,
            self._value_type,
        )

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base classes for iterators.
"""

from __future__ import annotations

import hashlib
from typing import Hashable

from .._bindings import Iterator, IteratorKind, IteratorState, Op
from .._caching import cache_with_registered_key_functions
from ..types import TypeDescriptor


class IteratorBase:
    """
    Iterators represent streams of data computed on the fly.

    Iterators can be used as inputs (or sometimes outputs) to most algorithms.
    """

    # Subclassing
    # -----------
    #
    # Subclasses must implement the following methods that return
    # Op objects.
    #
    # - _make_advance_op() -> Op
    # - _make_input_deref_op() -> Op | None
    # - _make_output_deref_op() -> Op | None
    #
    # Iterators composed of other iterators must also implement:
    #
    # - children property to return tuple of child iterators for dependency tracking
    #
    # Examples of such "compound" iterators include TransformIterator,
    # PermutationIterator, ReverseIterator and ZipIterator.
    #
    # The base class provides public cached accessors:
    #
    # - get_advance_op() -> Op (cached)
    # - get_input_deref_op() -> Op | None (cached)
    # - get_output_deref_op() -> Op | None (cached)

    __slots__ = [
        "_state_bytes",
        "_state_alignment",
        "_value_type",
        "_advance_op",
        "_input_deref_op",
        "_output_deref_op",
        "_uid_cached",
    ]

    def __init__(
        self,
        state_bytes: bytes,
        state_alignment: int,
        value_type: TypeDescriptor,
    ):
        """
        Args:
            state_bytes: bytes object representing iterator's state
            state_alignment: Alignment of the state
            value_type: Type of dereferenced values
        """
        self._state_bytes = state_bytes
        self._state_alignment = state_alignment
        self._value_type = value_type
        self._advance_op: Op | None = None
        self._input_deref_op: Op | None = None
        self._output_deref_op: Op | None = None
        self._uid_cached: str | None = None

    @property
    def state(self) -> IteratorState:
        """Return the iterator state for CCCL interop."""
        return IteratorState(self._state_bytes)

    @property
    def state_alignment(self) -> int:
        """Return the alignment of the iterator state."""
        return self._state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        """Return the TypeDescriptor for dereferenced values."""
        return self._value_type

    @property
    def children(self) -> tuple["IteratorBase", ...]:
        """Return child iterators for automatic dependency tracking. Override in subclasses."""
        return ()

    def _get_uid(self) -> str:
        """Return a deterministic unique identifier for this iterator type."""
        if self._uid_cached is None:
            self._uid_cached = _deterministic_suffix(self.kind)
        return self._uid_cached

    def _make_advance_symbol(self) -> str:
        """Generate symbol name for advance operation."""
        return f"{self.__class__.__name__}_advance_{self._get_uid()}"

    def _make_input_deref_symbol(self) -> str:
        """Generate symbol name for input dereference operation."""
        return f"{self.__class__.__name__}_input_deref_{self._get_uid()}"

    def _make_output_deref_symbol(self) -> str:
        """Generate symbol name for output dereference operation."""
        return f"{self.__class__.__name__}_output_deref_{self._get_uid()}"

    def get_advance_op(self) -> Op:
        """Get the cached Op for the advance operation."""
        if self._advance_op is None:
            self._advance_op = self._make_advance_op()
        return self._advance_op

    def get_input_deref_op(self) -> Op | None:
        """Get the cached Op for input dereference operation, or None if not supported."""
        if self._input_deref_op is None:
            self._input_deref_op = self._make_input_deref_op()
        return self._input_deref_op

    def get_output_deref_op(self) -> Op | None:
        """Get the cached Op for output dereference operation, or None if not supported."""
        if self._output_deref_op is None:
            self._output_deref_op = self._make_output_deref_op()
        return self._output_deref_op

    @property
    def is_input_iterator(self) -> bool:
        """Return True if this iterator supports input dereference."""
        return self.get_input_deref_op() is not None

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        return self.get_output_deref_op() is not None

    def to_cccl_iter(self, is_output: bool = False) -> Iterator:
        """
        Convert this iterator to a CCCL Iterator for algorithm interop.

        Args:
            is_output: If True, use output_dereference; otherwise use input_dereference

        Returns:
            CCCL Iterator object
        """
        # Get advance op
        advance_op = self.get_advance_op()

        # Get dereference op based on direction
        if is_output:
            deref_op = self.get_output_deref_op()
            if deref_op is None:
                raise ValueError("This iterator does not support output operations")
        else:
            deref_op = self.get_input_deref_op()
            if deref_op is None:
                raise ValueError("This iterator does not support input operations")

        # Create the CCCL Iterator
        return Iterator(
            self._state_alignment,
            IteratorKind.ITERATOR,
            advance_op,
            deref_op,
            self._value_type.info,
            state=self.state,
        )

    @property
    def kind(self) -> Hashable:
        """Return a hashable kind for caching purposes.

        Note: state_bytes is intentionally excluded - iterators with the same
        type structure but different runtime state should share cached reducers.
        """
        return (type(self).__name__, self._value_type)

    # Abstract methods for subclasses
    def _make_advance_op(self) -> Op:
        """
        Create Op object for advance operation.

        Returns:
            Op object with compiled LTOIR

        Subclasses can use any compilation approach:
        - compile_cpp_source_to_ltoir() for C++ source, then construct Op
        - Pre-compiled LTOIR bytes, then construct Op
        - Other compilation backends
        """
        raise NotImplementedError

    def _make_input_deref_op(self) -> Op | None:
        """
        Create Op object for input dereference operation.

        Returns:
            Op object with compiled LTOIR, or None if not supported

        Subclasses can use any compilation approach:
        - compile_cpp_source_to_ltoir() for C++ source, then construct Op
        - Pre-compiled LTOIR bytes, then construct Op
        - Other compilation backends
        """
        raise NotImplementedError

    def _make_output_deref_op(self) -> Op | None:
        """
        Create Op object for output dereference operation.

        Returns:
            Op object with compiled LTOIR, or None if not supported

        Subclasses can use any compilation approach:
        - compile_cpp_source_to_ltoir() for C++ source, then construct Op
        - Pre-compiled LTOIR bytes, then construct Op
        - Other compilation backends
        """
        raise NotImplementedError


def _deterministic_suffix(kind: Hashable) -> str:
    kind_str = str(kind)
    return hashlib.sha256(kind_str.encode()).hexdigest()[:16]


def compose_iterator_states(
    iterators: list[IteratorBase],
) -> tuple[bytes, int, list[int]]:
    """
    Concatenate multiple iterator states with proper alignment.

    This is used by composite iterators (like ZipIterator and PermutationIterator)
    that need to store multiple child iterator states in their own state.

    Args:
        iterators: List of child iterators whose states should be composed

    Returns:
        Tuple of:
        - combined_state_bytes: Concatenated state bytes with padding
        - combined_alignment: Maximum alignment requirement
        - offsets: List of byte offsets for each iterator's state
    """
    if not iterators:
        return (b"", 1, [])

    states = [bytes(memoryview(it.state)) for it in iterators]
    alignments = [it.state_alignment for it in iterators]

    offsets = []
    current_offset = 0
    combined = b""

    for state, align in zip(states, alignments):
        # Add padding to meet alignment requirement
        padding = (align - (current_offset % align)) % align
        combined += b"\x00" * padding
        current_offset += padding

        offsets.append(current_offset)
        combined += state
        current_offset += len(state)

    max_alignment = max(alignments)
    return (combined, max_alignment, offsets)


cache_with_registered_key_functions.register(IteratorBase, lambda it: it.kind)

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base iterator protocol for cuda.compute.

This module defines the IteratorProtocol that all iterators must implement,
whether they are Numba-based or pre-compiled (BYOC). This module has no
Numba dependencies.
"""

from typing import Any, Protocol, runtime_checkable


class IteratorKind:
    """
    Metadata about an iterator, analogous to the dtype of a NumPy array.

    The kind encapsulates the value_type (type of dereferenced values) and
    state_type (type of the iterator's internal state).
    """

    __slots__ = ["value_type", "state_type"]

    def __init__(self, value_type, state_type):
        """
        Create an IteratorKind.

        Args:
            value_type: The type of values produced by dereferencing the iterator.
                        Can be a Numba type, TypeDescriptor, or other type representation.
            state_type: The type of the iterator's internal state.
        """
        self.value_type = value_type
        self.state_type = state_type

    def __repr__(self):
        return (
            f"{self.__class__.__name__}[{str(self.value_type)}, {str(self.state_type)}]"
        )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.value_type == other.value_type
            and self.state_type == other.state_type
        )

    def __hash__(self):
        return hash((type(self), self.value_type, self.state_type))


@runtime_checkable
class IteratorProtocol(Protocol):
    """
    Protocol defining what an iterator must provide.

    All iterators (Numba-based or pre-compiled) must implement this protocol
    to be usable with cuda.compute algorithms.
    """

    @property
    def kind(self) -> IteratorKind:
        """Return the IteratorKind metadata for this iterator."""
        ...

    @property
    def value_type(self) -> Any:
        """Return the type of values produced by dereferencing this iterator."""
        ...

    def to_cccl_iter(self, is_output: bool) -> Any:
        """
        Convert this iterator to a CCCL Iterator object for C++ interop.

        Args:
            is_output: True if this iterator is used for output, False for input.

        Returns:
            A CCCL Iterator object.
        """
        ...


__all__ = [
    "IteratorKind",
    "IteratorProtocol",
]

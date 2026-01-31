# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ReverseIterator implementation."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from .._bindings import IteratorState
from .._utils.protocols import get_size
from ..types import TypeDescriptor
from ._base import IteratorBase
from ._codegen_utils import (
    collect_child_ltoirs,
    compile_cpp_source_to_ltoir,
    format_advance,
    format_input_dereference,
    format_output_dereference,
)

if TYPE_CHECKING:
    pass


def _ensure_iterator(obj):
    """Wrap array in PointerIterator at the END of the array for reverse iteration."""
    from ..typing import DeviceArrayLike
    from ._pointer import PointerIterator

    if isinstance(obj, PointerIterator):
        array_len = get_size(obj._array)
        return obj + (array_len - 1)
    if isinstance(obj, DeviceArrayLike):
        array_len = get_size(obj)
        return PointerIterator(obj) + (array_len - 1)
    if isinstance(obj, IteratorBase):
        return obj

    raise TypeError("ReverseIterator requires a device array or iterator")


class ReverseIterator(IteratorBase):
    """
    Iterator that reverses the direction of an underlying iterator.

    Advance with positive offset moves backward in the underlying iterator.
    """

    __slots__ = [
        "_underlying",
    ]

    def __init__(self, underlying):
        """
        Create a reverse iterator.

        Args:
            underlying: The underlying iterator or array to reverse
        """
        self._underlying = _ensure_iterator(underlying)

        super().__init__(
            state_bytes=bytes(self._underlying.state),
            state_alignment=self._underlying.state_alignment,
            value_type=self._underlying.value_type,
        )

    def _provide_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        """Provide compiled LTOIR for advance that negates offset direction."""
        underlying_advance, _, _ = self._underlying.get_advance_ltoir()
        symbol = self._make_advance_symbol()

        body = dedent(f"""
            int64_t neg_offset = -static_cast<int64_t>(*static_cast<uint64_t*>(offset));
            {underlying_advance}(state, &neg_offset);
        """).strip()

        source = format_advance(symbol, body, extern_symbols=[underlying_advance])
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs([self._underlying], "advance")
        return (symbol, ltoir, child_ltoirs)

    def _provide_input_deref_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Provide compiled LTOIR for input dereference that delegates to underlying."""
        underlying_result = self._underlying.get_input_dereference_ltoir()
        if underlying_result is None:
            return None
        underlying_deref, _, _ = underlying_result

        symbol = self._make_input_deref_symbol()

        body = dedent(f"""
            {underlying_deref}(state, result);
        """).strip()

        source = format_input_dereference(
            symbol, body, extern_symbols=[underlying_deref]
        )
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs([self._underlying], "input_deref")
        return (symbol, ltoir, child_ltoirs)

    def _provide_output_deref_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Provide compiled LTOIR for output dereference that delegates to underlying."""
        underlying_result = self._underlying.get_output_dereference_ltoir()
        if underlying_result is None:
            return None
        underlying_deref, _, _ = underlying_result

        symbol = self._make_output_deref_symbol()

        body = dedent(f"""
            {underlying_deref}(state, value);
        """).strip()

        source = format_output_dereference(
            symbol, body, extern_symbols=[underlying_deref]
        )
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs([self._underlying], "output_deref")
        return (symbol, ltoir, child_ltoirs)

    @property
    def state(self) -> IteratorState:
        return self._underlying.state

    @property
    def state_alignment(self) -> int:
        return self._underlying.state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        return self._underlying.value_type

    @property
    def children(self):
        return (self._underlying,)

    @property
    def is_input_iterator(self) -> bool:
        return self._underlying.is_input_iterator

    @property
    def is_output_iterator(self) -> bool:
        return self._underlying.is_output_iterator

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ReverseIterator", self._underlying.kind)

    def __add__(self, offset: int):
        return ReverseIterator(self._underlying + offset)

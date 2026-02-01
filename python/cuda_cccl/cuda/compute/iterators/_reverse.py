# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ReverseIterator implementation."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from .._bindings import Op, OpKind
from .._utils.protocols import get_size
from ._base import IteratorBase
from ._codegen_utils import (
    collect_child_ltoirs,
    collect_child_op_names,
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

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that negates offset direction."""
        advance_names = collect_child_op_names([self._underlying], "advance")
        underlying_advance = advance_names[0]
        symbol = self._make_advance_symbol()

        body = dedent(f"""
            int64_t neg_offset = -static_cast<int64_t>(*static_cast<uint64_t*>(offset));
            {underlying_advance}(state, &neg_offset);
        """).strip()

        source = format_advance(symbol, body, extern_symbols=[underlying_advance])
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs([self._underlying], "advance")

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs if child_ltoirs else None,
        )

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input dereference that delegates to underlying."""
        if self._underlying.get_input_deref_op() is None:
            return None

        deref_names = collect_child_op_names([self._underlying], "input_deref")
        underlying_deref = deref_names[0]
        symbol = self._make_input_deref_symbol()

        body = dedent(f"""
            {underlying_deref}(state, result);
        """).strip()

        source = format_input_dereference(
            symbol, body, extern_symbols=[underlying_deref]
        )
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs([self._underlying], "input_deref")

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs if child_ltoirs else None,
        )

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output dereference that delegates to underlying."""
        if self._underlying.get_output_deref_op() is None:
            return None

        deref_names = collect_child_op_names([self._underlying], "output_deref")
        underlying_deref = deref_names[0]
        symbol = self._make_output_deref_symbol()

        body = dedent(f"""
            {underlying_deref}(state, value);
        """).strip()

        source = format_output_dereference(
            symbol, body, extern_symbols=[underlying_deref]
        )
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs([self._underlying], "output_deref")

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs if child_ltoirs else None,
        )

    @property
    def children(self):
        return (self._underlying,)

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ReverseIterator", self._underlying.kind)

    def __add__(self, offset: int):
        return ReverseIterator(self._underlying + offset)

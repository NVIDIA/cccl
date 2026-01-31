# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PermutationIterator implementation."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from .._bindings import Op, OpKind
from .._cpp_codegen import cpp_type_from_descriptor
from ._base import IteratorBase
from ._codegen_utils import (
    collect_child_ltoirs,
    collect_child_op_names,
    compile_cpp_source_to_ltoir,
    compose_iterator_states,
    format_advance,
    format_input_dereference,
    format_output_dereference,
)

if TYPE_CHECKING:
    pass


def _ensure_iterator(obj):
    """Wrap array in PointerIterator if needed."""
    from ._pointer import PointerIterator

    if isinstance(obj, IteratorBase):
        return obj
    if hasattr(obj, "__cuda_array_interface__"):
        return PointerIterator(obj)
    raise TypeError("PermutationIterator requires iterators or device arrays")


class PermutationIterator(IteratorBase):
    """
    Iterator that accesses values through an index mapping.

    At position i, yields values[indices[i]]. The indices iterator is advanced
    normally, and the values iterator is accessed via random access using the
    index value.
    """

    __slots__ = [
        "_values",
        "_indices",
        "_values_offset",
        "_indices_offset",
    ]

    def __init__(
        self,
        values,
        indices,
    ):
        """
        Create a permutation iterator.

        Args:
            values: Iterator or array providing the values to be permuted
            indices: Iterator or array providing the indices for permutation
        """
        # Wrap arrays in PointerIterator
        self._values = _ensure_iterator(values)
        self._indices = _ensure_iterator(indices)

        # Compose states from both iterators
        state_bytes, state_alignment, offsets = compose_iterator_states(
            [self._values, self._indices]
        )
        self._values_offset = offsets[0]
        self._indices_offset = offsets[1]

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=state_alignment,
            value_type=self._values.value_type,
        )

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that only advances indices iterator."""
        advance_names = collect_child_op_names([self._indices], "advance")
        indices_advance = advance_names[0]
        symbol = self._make_advance_symbol()

        body = dedent(f"""
            char* indices_state = static_cast<char*>(state) + {self._indices_offset};
            {indices_advance}(indices_state, offset);
        """).strip()

        source = format_advance(symbol, body, extern_symbols=[indices_advance])
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs([self._indices], "advance")

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs if child_ltoirs else None,
        )

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input deref that reads index then accesses values."""
        if self._indices.get_input_deref_op() is None:
            raise ValueError("Indices iterator must support input dereference")

        if self._values.get_input_deref_op() is None:
            return None

        deref_names = collect_child_op_names(
            [self._indices, self._values], "input_deref"
        )
        indices_deref = deref_names[0]
        values_deref = deref_names[1]

        # Also need values advance for random access
        advance_names = collect_child_op_names([self._values], "advance")
        values_advance = advance_names[0]

        symbol = self._make_input_deref_symbol()
        idx_type = cpp_type_from_descriptor(self._indices.value_type)
        values_state_size = len(bytes(memoryview(self._values.state)))

        body = dedent(f"""
            char* values_state = static_cast<char*>(state) + {self._values_offset};
            char* indices_state = static_cast<char*>(state) + {self._indices_offset};

            alignas({self._indices.value_type.alignment}) {idx_type} idx;
            {indices_deref}(indices_state, &idx);

            alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
            memcpy(temp_values, values_state, {values_state_size});

            uint64_t offset = static_cast<uint64_t>(idx);
            {values_advance}(temp_values, &offset);
            {values_deref}(temp_values, result);
        """).strip()

        source = format_input_dereference(
            symbol, body, extern_symbols=[indices_deref, values_advance, values_deref]
        )
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        # Include values.advance LTOIR and child LTOIRs from both iterators
        deref_ltoirs = collect_child_ltoirs(
            [self._indices, self._values], "input_deref"
        )
        advance_ltoirs = collect_child_ltoirs([self._values], "advance")

        extra_ltoirs = advance_ltoirs + deref_ltoirs
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=extra_ltoirs if extra_ltoirs else None,
        )

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output deref that reads index then writes to values."""
        if self._indices.get_input_deref_op() is None:
            raise ValueError("Indices iterator must support input dereference")

        if self._values.get_output_deref_op() is None:
            return None

        # Get indices input_deref and values output_deref names
        indices_deref_names = collect_child_op_names([self._indices], "input_deref")
        values_deref_names = collect_child_op_names([self._values], "output_deref")
        indices_deref = indices_deref_names[0]
        values_deref = values_deref_names[0]

        # Also need values advance for random access
        advance_names = collect_child_op_names([self._values], "advance")
        values_advance = advance_names[0]

        symbol = self._make_output_deref_symbol()
        idx_type = cpp_type_from_descriptor(self._indices.value_type)
        values_state_size = len(bytes(memoryview(self._values.state)))

        body = dedent(f"""
            char* values_state = static_cast<char*>(state) + {self._values_offset};
            char* indices_state = static_cast<char*>(state) + {self._indices_offset};

            alignas({self._indices.value_type.alignment}) {idx_type} idx;
            {indices_deref}(indices_state, &idx);

            alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
            memcpy(temp_values, values_state, {values_state_size});

            uint64_t offset = static_cast<uint64_t>(idx);
            {values_advance}(temp_values, &offset);
            {values_deref}(temp_values, value);
        """).strip()

        source = format_output_dereference(
            symbol, body, extern_symbols=[indices_deref, values_advance, values_deref]
        )
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        # Include values.advance LTOIR and child LTOIRs from both iterators
        # For output_deref, we need indices input_deref and values output_deref
        indices_deref_ltoirs = collect_child_ltoirs([self._indices], "input_deref")
        values_deref_ltoirs = collect_child_ltoirs([self._values], "output_deref")
        advance_ltoirs = collect_child_ltoirs([self._values], "advance")

        extra_ltoirs = advance_ltoirs + indices_deref_ltoirs + values_deref_ltoirs
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=extra_ltoirs if extra_ltoirs else None,
        )

    @property
    def children(self):
        return (self._values, self._indices)

    @property
    def is_input_iterator(self) -> bool:
        return self._values.is_input_iterator

    @property
    def is_output_iterator(self) -> bool:
        return self._values.is_output_iterator

    def __add__(self, offset: int) -> "PermutationIterator":
        """Advance the indices iterator by offset, keeping values at base."""
        return PermutationIterator(
            self._values,  # values stays at base for random access
            self._indices + offset,  # only indices advances  # type: ignore[operator]
        )

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("PermutationIterator", self._values.kind, self._indices.kind)

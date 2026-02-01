# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ZipIterator implementation."""

from __future__ import annotations

from .._bindings import Op, OpKind
from ..types import struct
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


def _generate_advance_body(child_symbols: list[str], state_offsets: list[int]) -> str:
    """
    Generate C++ code to call advance on multiple child iterators.

    Args:
        child_symbols: List of child iterator advance function names
        state_offsets: List of byte offsets for each child's state

    Returns:
        C++ code calling each child's advance function
    """
    return "\n".join(
        f"{sym}(static_cast<char*>(state) + {off}, offset);"
        for sym, off in zip(child_symbols, state_offsets)
    )


def _generate_deref_body(
    child_symbols: list[str],
    state_offsets: list[int],
    value_offsets: list[int],
    result_name: str = "result",
) -> str:
    """
    Generate C++ code to call dereference on multiple child iterators.

    Used by ZipIterator to read/write to multiple child iterators at once.

    Args:
        child_symbols: List of child iterator dereference function names
        state_offsets: List of byte offsets for each child's state
        value_offsets: List of byte offsets for each child's value in result
        result_name: Name of result/value parameter (usually "result" or "value")

    Returns:
        C++ code calling each child's dereference function
    """
    return "\n".join(
        f"{sym}(static_cast<char*>(state) + {state_off}, "
        f"static_cast<char*>({result_name}) + {val_off});"
        for sym, state_off, val_off in zip(child_symbols, state_offsets, value_offsets)
    )


def _ensure_iterator(obj):
    """Wrap array in PointerIterator if needed."""
    from ._pointer import PointerIterator

    if isinstance(obj, IteratorBase):
        return obj
    if hasattr(obj, "__cuda_array_interface__"):
        return PointerIterator(obj)
    raise TypeError("ZipIterator requires iterators or device arrays")


class ZipIterator(IteratorBase):
    """
    Iterator that zips multiple iterators together.

    At each position, yields a tuple/struct of values from all underlying iterators.
    """

    __slots__ = [
        "_iterators",
        # TypeDescriptors for each component (for Numba tuple)
        "_field_names",
        "_value_offsets",
        "_state_offsets",
        "_advance_result",
        "_input_deref_result",
        "_output_deref_result",
    ]

    def __init__(self, *args):
        """
        Create a zip iterator.

        Args:
            *args: Iterators or arrays to zip together. Can be:
                   - Multiple iterators/arrays: ZipIterator(it1, it2, it3)
                   - A single sequence of iterators: ZipIterator([it1, it2, it3])
        """
        # Handle both ZipIterator(it1, it2) and ZipIterator([it1, it2])
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            iterators = args[0]
        else:
            iterators = args

        if len(iterators) < 1:
            raise ValueError("ZipIterator requires at least one iterator")

        # Wrap arrays in PointerIterator
        iterators = [_ensure_iterator(it) for it in iterators]

        self._iterators = list(iterators)

        # Compose states from all iterators
        self._state_bytes, self._state_alignment, self._state_offsets = (
            compose_iterator_states(self._iterators)
        )

        # Build combined value type (struct layout)
        self._field_names = [f"field_{i}" for i in range(len(self._iterators))]
        fields = {
            name: it.value_type for name, it in zip(self._field_names, self._iterators)
        }
        self._value_type = struct(fields, name=f"Zip{len(iterators)}")
        self._value_offsets = [
            self._value_type.dtype.fields[name][1] for name in self._field_names
        ]

        super().__init__(
            state_bytes=self._state_bytes,
            state_alignment=self._state_alignment,
            value_type=self._value_type,
        )

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that calls all child iterator advances."""
        advance_names = collect_child_op_names(self._iterators, "advance")
        symbol = self._make_advance_symbol()

        body = _generate_advance_body(advance_names, self._state_offsets)

        source = format_advance(symbol, body, extern_symbols=advance_names)
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs(self._iterators, "advance")

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs if child_ltoirs else None,
        )

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input deref that calls all child iterator input derefs."""
        # Check if all iterators support input dereference
        if not all(it.get_input_deref_op() is not None for it in self._iterators):
            return None

        deref_names = collect_child_op_names(self._iterators, "input_deref")
        symbol = self._make_input_deref_symbol()

        body = _generate_deref_body(
            deref_names, self._state_offsets, self._value_offsets, result_name="result"
        )

        source = format_input_dereference(symbol, body, extern_symbols=deref_names)
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs(self._iterators, "input_deref")

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs if child_ltoirs else None,
        )

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output deref that calls all child iterator output derefs."""
        # Check if all iterators support output dereference
        if not all(it.get_output_deref_op() is not None for it in self._iterators):
            return None

        deref_names = collect_child_op_names(self._iterators, "output_deref")
        symbol = self._make_output_deref_symbol()

        body = _generate_deref_body(
            deref_names, self._state_offsets, self._value_offsets, result_name="value"
        )

        source = format_output_dereference(symbol, body, extern_symbols=deref_names)
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        child_ltoirs = collect_child_ltoirs(self._iterators, "output_deref")

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs if child_ltoirs else None,
        )

    @property
    def children(self):
        return tuple(self._iterators)

    def __add__(self, offset: int) -> "ZipIterator":
        """Advance all child iterators by offset."""
        advanced_iterators = [it + offset for it in self._iterators]  # type: ignore[operator]
        return ZipIterator(*advanced_iterators)

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ZipIterator", tuple(it.kind for it in self._iterators))

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ReverseIterator implementation."""

from __future__ import annotations

from textwrap import dedent

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir
from .._utils.protocols import get_size, is_device_array
from ._base import IteratorBase
from ._common import CUDA_PREAMBLE, ensure_iterator


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

        if is_device_array(underlying):
            # TODO: this is probably incorrect behaviour. In C++, initializing
            # with a pointer to the end of the array is left to be done explicitly
            # by the user.
            self._underlying = ensure_iterator(underlying) + (get_size(underlying) - 1)
        else:
            self._underlying = ensure_iterator(underlying)

        super().__init__(
            state_bytes=bytes(self._underlying.state),
            state_alignment=self._underlying.state_alignment,
            value_type=self._underlying.value_type,
        )

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that negates offset direction."""
        child_op = self._underlying.get_advance_op()
        symbol = self._make_advance_symbol()

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* offset);

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                int64_t neg_offset = -static_cast<int64_t>(*static_cast<uint64_t*>(offset));
                {child_op.name}(state, &neg_offset);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[child_op.ltoir, *child_op.extra_ltoirs],
        )

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input dereference that delegates to underlying."""
        child_op = self._underlying.get_input_deref_op()
        if child_op is None:
            return None

        symbol = self._make_input_deref_symbol()

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* result);

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                {child_op.name}(state, result);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[child_op.ltoir, *child_op.extra_ltoirs],
        )

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output dereference that delegates to underlying."""
        child_op = self._underlying.get_output_deref_op()
        if child_op is None:
            return None

        symbol = self._make_output_deref_symbol()

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* value);

            extern "C" __device__ void {symbol}(void* state, void* value) {{
                {child_op.name}(state, value);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[child_op.ltoir, *child_op.extra_ltoirs],
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

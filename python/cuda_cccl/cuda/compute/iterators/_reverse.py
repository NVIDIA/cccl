# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ReverseIterator implementation."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir
from .._utils.protocols import get_size, is_device_array
from ._base import IteratorBase

CUDA_PREAMBLE = """#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda/std/cstring>
using namespace cuda::std;
"""

if TYPE_CHECKING:
    pass


def _ensure_iterator(obj):
    """Wrap array in PointerIterator at the END of the array for reverse iteration."""
    from ._pointer import PointerIterator

    if isinstance(obj, PointerIterator):
        array_len = get_size(obj._array)
        return obj + (array_len - 1)
    if is_device_array(obj):
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

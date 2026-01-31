# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CountingIterator implementation."""

from __future__ import annotations

from textwrap import dedent

import numpy as np

from .._bindings import Op, OpKind
from .._cpp_codegen import cpp_type_from_descriptor
from ..types import from_numpy_dtype
from ._base import IteratorBase
from ._codegen_utils import (
    compile_cpp_source_to_ltoir,
    format_advance,
    format_input_dereference,
)


class CountingIterator(IteratorBase):
    """
    Iterator representing a sequence of incrementing values.

    The iterator starts at `start` and increments by 1 for each advance.
    """

    def __init__(self, start: np.number):
        """
        Create a counting iterator starting at `start`.

        Args:
            start: The initial value (must be a numpy scalar)
        """
        if not isinstance(start, np.generic):
            start = np.array(start).flatten()[0]

        self._start_value = start
        value_type = from_numpy_dtype(start.dtype)
        state_bytes = start.tobytes()

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

    def _make_advance_op(self) -> Op:
        symbol = self._make_advance_symbol()
        cpp_type = cpp_type_from_descriptor(self._value_type)

        body = dedent(f"""
            auto* s = static_cast<{cpp_type}*>(state);
            auto dist = *static_cast<uint64_t*>(offset);
            *s += static_cast<{cpp_type}>(dist);
        """).strip()

        source = format_advance(symbol, body)
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=None,
        )

    def _make_input_deref_op(self) -> Op | None:
        symbol = self._make_input_deref_symbol()
        cpp_type = cpp_type_from_descriptor(self._value_type)

        body = dedent(f"""
            *static_cast<{cpp_type}*>(result) = *static_cast<{cpp_type}*>(state);
        """).strip()

        source = format_input_dereference(symbol, body)
        ltoir = compile_cpp_source_to_ltoir(source, symbol)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=None,
        )

    def _make_output_deref_op(self) -> Op | None:
        return None

    def __add__(self, offset: int) -> "CountingIterator":
        """Return a new CountingIterator advanced by offset elements."""
        new_start = self._start_value + offset
        return CountingIterator(new_start)

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CountingIterator implementation."""

from __future__ import annotations

from textwrap import dedent

import numpy as np

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir, cpp_type_from_descriptor
from ..types import from_numpy_dtype
from ._base import IteratorBase
from ._common import CUDA_PREAMBLE


class CountingIterator(IteratorBase):
    """
    Iterator representing a sequence of incrementing values.

    Similar to `thrust::counting_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1counting__iterator.html>`_.

    The iterator starts at `start` and increments by 1 for each advance.

    Example:
        The code snippet below demonstrates the usage of a ``CountingIterator``
        representing the sequence ``[10, 11, 12]``:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/counting_iterator_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        start: The initial value of the sequence
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

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                auto* s = static_cast<{cpp_type}*>(state);
                auto dist = *static_cast<uint64_t*>(offset);
                *s += static_cast<{cpp_type}>(dist);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[],
        )

    def _make_input_deref_op(self) -> Op | None:
        symbol = self._make_input_deref_symbol()
        cpp_type = cpp_type_from_descriptor(self._value_type)

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                *static_cast<{cpp_type}*>(result) = *static_cast<{cpp_type}*>(state);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[],
        )

    def _make_output_deref_op(self) -> Op | None:
        return None

    def __add__(self, offset: int) -> "CountingIterator":
        """Return a new CountingIterator advanced by offset elements."""
        new_start = self._start_value + offset
        return CountingIterator(new_start)

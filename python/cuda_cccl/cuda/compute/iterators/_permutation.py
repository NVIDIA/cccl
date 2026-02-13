# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PermutationIterator implementation."""

from __future__ import annotations

from textwrap import dedent

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir, make_variable_declaration
from ._base import IteratorBase, compose_iterator_states
from ._common import CUDA_PREAMBLE, ensure_iterator


class PermutationIterator(IteratorBase):
    """
    Iterator that accesses values through an index mapping.

    At position i, yields values[indices[i]].

    Similar to `thrust::permutation_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1permutation__iterator.html>`_.

    Example:
        The code snippet below demonstrates accessing values through an index mapping.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/permutation_iterator_basic.py
            :language: python
            :start-after: # example-begin
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
        self._values = ensure_iterator(values)
        self._indices = ensure_iterator(indices)

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
        child_op = self._indices.get_advance_op()
        symbol = self._make_advance_symbol()

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* offset);

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                char* indices_state = static_cast<char*>(state) + {self._indices_offset};
                {child_op.name}(indices_state, offset);
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
        """Provide Op for input deref that reads index then accesses values."""
        indices_deref_op = self._indices.get_input_deref_op()
        if indices_deref_op is None:
            raise ValueError("Indices iterator must support input dereference")

        values_deref_op = self._values.get_input_deref_op()
        if values_deref_op is None:
            return None

        # Also need values advance for random access
        values_advance_op = self._values.get_advance_op()

        symbol = self._make_input_deref_symbol()
        idx_decl = make_variable_declaration(self._indices.value_type, "idx")
        values_state_size = len(bytes(memoryview(self._values.state)))

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {indices_deref_op.name}(void* state, void* result);
            extern "C" __device__ void {values_advance_op.name}(void* state, void* offset);
            extern "C" __device__ void {values_deref_op.name}(void* state, void* result);

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                char* values_state = static_cast<char*>(state) + {self._values_offset};
                char* indices_state = static_cast<char*>(state) + {self._indices_offset};

                {idx_decl}
                {indices_deref_op.name}(indices_state, &idx);

                alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
                memcpy(temp_values, values_state, {values_state_size});

                uint64_t offset = static_cast<uint64_t>(idx);
                {values_advance_op.name}(temp_values, &offset);
                {values_deref_op.name}(temp_values, result);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[
                values_advance_op.ltoir,
                *values_advance_op.extra_ltoirs,
                indices_deref_op.ltoir,
                *indices_deref_op.extra_ltoirs,
                values_deref_op.ltoir,
                *values_deref_op.extra_ltoirs,
            ],
        )

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output deref that reads index then writes to values."""
        indices_deref_op = self._indices.get_input_deref_op()
        if indices_deref_op is None:
            raise ValueError("Indices iterator must support input dereference")

        values_deref_op = self._values.get_output_deref_op()
        if values_deref_op is None:
            return None

        # Also need values advance for random access
        values_advance_op = self._values.get_advance_op()

        symbol = self._make_output_deref_symbol()
        idx_decl = make_variable_declaration(self._indices.value_type, "idx")
        values_state_size = len(bytes(memoryview(self._values.state)))

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {indices_deref_op.name}(void* state, void* result);
            extern "C" __device__ void {values_advance_op.name}(void* state, void* offset);
            extern "C" __device__ void {values_deref_op.name}(void* state, void* value);

            extern "C" __device__ void {symbol}(void* state, void* value) {{
                char* values_state = static_cast<char*>(state) + {self._values_offset};
                char* indices_state = static_cast<char*>(state) + {self._indices_offset};

                {idx_decl}
                {indices_deref_op.name}(indices_state, &idx);

                alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
                memcpy(temp_values, values_state, {values_state_size});

                uint64_t offset = static_cast<uint64_t>(idx);
                {values_advance_op.name}(temp_values, &offset);
                {values_deref_op.name}(temp_values, value);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[
                values_advance_op.ltoir,
                *values_advance_op.extra_ltoirs,
                indices_deref_op.ltoir,
                *indices_deref_op.extra_ltoirs,
                values_deref_op.ltoir,
                *values_deref_op.extra_ltoirs,
            ],
        )

    @property
    def children(self):
        return (self._values, self._indices)

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

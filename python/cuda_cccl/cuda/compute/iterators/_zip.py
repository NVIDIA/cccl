# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ZipIterator implementation."""

from __future__ import annotations

from textwrap import dedent

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir
from ..types import struct
from ._base import IteratorBase, compose_iterator_states
from ._common import CUDA_PREAMBLE, ensure_iterator


class ZipIterator(IteratorBase):
    """
    Iterator that zips multiple iterators together.

    At each position, yields a tuple of values from all underlying iterators.

    Similar to `thrust::zip_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1zip__iterator.html>`_.

    Example:
        The code snippet below demonstrates how to zip together an array and a
        :class:`CountingIterator <cuda.compute.iterators.CountingIterator>` to
        find the index of the maximum value of the array.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/zip_iterator_counting.py
            :language: python
            :start-after: # example-begin
    """

    __slots__ = [
        "_iterators",
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
        iterators = [ensure_iterator(it) for it in iterators]

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
        child_ops = [it.get_advance_op() for it in self._iterators]
        symbol = self._make_advance_symbol()

        externs = "\n".join(
            f'extern "C" __device__ void {op.name}(void* state, void* offset);'
            for op in child_ops
        )

        calls = "\n        ".join(
            f"{op.name}(static_cast<char*>(state) + {offset}, offset);"
            for op, offset in zip(child_ops, self._state_offsets)
        )

        source = dedent(f"""
            {CUDA_PREAMBLE}

            {externs}

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                {calls}
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[
                ltoir for op in child_ops for ltoir in [op.ltoir, *op.extra_ltoirs]
            ],
        )

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input deref that calls all child iterator input derefs."""
        child_ops = [it.get_input_deref_op() for it in self._iterators]
        if not all(op is not None for op in child_ops):
            return None

        symbol = self._make_input_deref_symbol()

        externs = "\n".join(
            f'extern "C" __device__ void {op.name}(void* state, void* result);'
            for op in child_ops
        )

        calls = "\n        ".join(
            f"{op.name}(static_cast<char*>(state) + {state_off}, "
            f"static_cast<char*>(result) + {val_off});"
            for op, state_off, val_off in zip(
                child_ops, self._state_offsets, self._value_offsets
            )
        )

        source = dedent(f"""
            {CUDA_PREAMBLE}

            {externs}

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                {calls}
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[
                ltoir for op in child_ops for ltoir in [op.ltoir, *op.extra_ltoirs]
            ],
        )

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output deref that calls all child iterator output derefs."""
        child_ops = [it.get_output_deref_op() for it in self._iterators]
        if not all(op is not None for op in child_ops):
            return None

        symbol = self._make_output_deref_symbol()

        externs = "\n".join(
            f'extern "C" __device__ void {op.name}(void* state, void* value);'
            for op in child_ops
        )

        calls = "\n        ".join(
            f"{op.name}(static_cast<char*>(state) + {state_off}, "
            f"static_cast<char*>(value) + {val_off});"
            for op, state_off, val_off in zip(
                child_ops, self._state_offsets, self._value_offsets
            )
        )

        source = dedent(f"""
            {CUDA_PREAMBLE}

            {externs}

            extern "C" __device__ void {symbol}(void* state, void* value) {{
                {calls}
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[
                ltoir for op in child_ops for ltoir in [op.ltoir, *op.extra_ltoirs]
            ],
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

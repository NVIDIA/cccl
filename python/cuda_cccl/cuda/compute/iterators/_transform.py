# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TransformIterator implementation."""

from __future__ import annotations

from textwrap import dedent

from .._bindings import Op, OpKind
from .._cpp_codegen import compile_cpp_to_ltoir, cpp_type_from_descriptor
from ..op import make_op_adapter
from ..types import TypeDescriptor
from ._base import IteratorBase

CUDA_PREAMBLE = """#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda/std/cstring>
using namespace cuda::std;
"""


class TransformIterator(IteratorBase):
    """
    Iterator that applies a transform operation to values from an underlying iterator.

    For input iteration: reads from underlying, applies transform, returns result.
    For output iteration: applies transform to input, writes to underlying.
    """

    __slots__ = [
        "_underlying",
        "_transform_op",
        "_value_type",
        "_is_input",
        "_compiled_op",
    ]

    def __init__(
        self,
        underlying: IteratorBase,
        transform_op,
        output_value_type: TypeDescriptor,
        is_input: bool = True,
    ):
        """
        Create a transform iterator.

        Args:
            underlying: The underlying iterator to transform
            transform_op: The unary transform operation (OpProtocol)
            output_value_type: TypeDescriptor for the transformed value type
            is_input: True for input iterator, False for output iterator
        """
        self._underlying = underlying
        self._transform_op = make_op_adapter(transform_op)
        self._value_type = output_value_type
        self._is_input = is_input
        self._compiled_op = None  # Lazy compiled Op

        super().__init__(
            state_bytes=bytes(self._underlying.state),
            state_alignment=self._underlying.state_alignment,
            value_type=output_value_type,
        )

    def _get_compiled_op(self):
        """Get the compiled Op, compiling lazily if needed."""
        if self._compiled_op is None:
            if self._is_input:
                input_type = self._underlying.value_type
                output_type = self._value_type
            else:
                input_type = self._value_type
                output_type = self._underlying.value_type

            self._compiled_op = self._transform_op.compile(
                (input_type,),
                output_type,
            )
        return self._compiled_op

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that delegates to underlying iterator."""
        child_op = self._underlying.get_advance_op()
        symbol = self._make_advance_symbol()

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* offset);

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                {child_op.name}(state, offset);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source, (symbol,))

        # Flatten child LTOIRs
        child_ltoirs = [child_op.ltoir]
        if child_op.extra_ltoirs:
            child_ltoirs.extend(child_op.extra_ltoirs)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs,
        )

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input dereference that reads from underlying then transforms."""
        if not self._is_input:
            return None

        child_op = self._underlying.get_input_deref_op()
        if child_op is None:
            raise ValueError("Underlying iterator must support input dereference")

        compiled_op = self._get_compiled_op()
        symbol = self._make_input_deref_symbol()
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* result);
            extern "C" __device__ void {compiled_op.name}(void* input, void* output);

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                {underlying_type} temp;
                {child_op.name}(state, &temp);
                {compiled_op.name}(&temp, result);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source, (symbol,))

        # Flatten child LTOIRs
        child_ltoirs = [compiled_op.ltoir]
        if compiled_op.extra_ltoirs:
            child_ltoirs.extend(compiled_op.extra_ltoirs)
        child_ltoirs.append(child_op.ltoir)
        if child_op.extra_ltoirs:
            child_ltoirs.extend(child_op.extra_ltoirs)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs,
        )

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output dereference that transforms then writes to underlying."""
        if self._is_input:
            return None

        child_op = self._underlying.get_output_deref_op()
        if child_op is None:
            raise ValueError("Underlying iterator must support output dereference")

        compiled_op = self._get_compiled_op()
        symbol = self._make_output_deref_symbol()
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* value);
            extern "C" __device__ void {compiled_op.name}(void* input, void* output);

            extern "C" __device__ void {symbol}(void* state, void* value) {{
                {underlying_type} temp;
                {compiled_op.name}(value, &temp);
                {child_op.name}(state, &temp);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source, (symbol,))

        # Flatten child LTOIRs
        child_ltoirs = [compiled_op.ltoir]
        if compiled_op.extra_ltoirs:
            child_ltoirs.extend(compiled_op.extra_ltoirs)
        child_ltoirs.append(child_op.ltoir)
        if child_op.extra_ltoirs:
            child_ltoirs.extend(child_op.extra_ltoirs)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=child_ltoirs,
        )

    def advance(self, offset: int) -> "TransformIterator":
        """Return a new iterator advanced by offset elements."""
        if not hasattr(self._underlying, "__add__"):
            raise AttributeError("Underlying iterator does not support advance")
        return TransformIterator(
            self._underlying + offset,  # type: ignore[operator, arg-type]
            self._transform_op,
            self._value_type,
            is_input=self._is_input,
        )

    def __add__(self, offset: int) -> "TransformIterator":
        return self.advance(offset)

    def __radd__(self, offset: int) -> "TransformIterator":
        return self.advance(offset)

    @property
    def children(self):
        return (self._underlying,)

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        # Convert _value_type to tuple if it's a list (for output iterators)
        value_type = (
            tuple(self._value_type)
            if isinstance(self._value_type, list)
            else self._value_type
        )
        return (
            "TransformIterator",
            self._is_input,
            self._transform_op,
            self._underlying.kind,
            value_type,
        )

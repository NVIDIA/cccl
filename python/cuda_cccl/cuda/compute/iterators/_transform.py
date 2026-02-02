# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TransformIterator implementation."""

from __future__ import annotations

from textwrap import dedent

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir, make_variable_declaration
from ..op import make_op_adapter
from ..types import TypeDescriptor, signature_from_annotations
from ._base import IteratorBase

CUDA_PREAMBLE = """#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda/std/cstring>
using namespace cuda::std;
"""


class TransformIterator(IteratorBase):
    """
    An iterator that applies a unary function to elements as they are read from an underlying iterator.

    Similar to `thrust::transform_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1transform__iterator.html>`_.

    For input iteration (default): reads from underlying, applies transform, returns result.
    For output iteration: applies transform to input values, writes to underlying.

    Example:
        The code snippet below demonstrates the usage of a ``TransformIterator`` composed with a ``CountingIterator``
        to transform the input before performing a reduction:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/transform_iterator_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        underlying: The underlying iterator or device array
        transform_op: The unary operation to apply
        output_value_type: TypeDescriptor for the output type (optional, will be inferred if not provided)
        is_input: True for input iterator (default), False for output iterator
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
        underlying,
        transform_op,
        value_type: TypeDescriptor | None = None,
        is_input: bool = True,
    ):
        """
        Create a transform iterator.

        Args:
            underlying: The underlying iterator or device array to transform
            transform_op: The unary transform operation (callable or OpKind)
            value_type: TypeDescriptor for the transformed value type.
                            For input iterators: inferred if None.
                            For output iterators: must be provided or have annotations.
            is_input: True for input iterator, False for output iterator
        """
        # Wrap device arrays in PointerIterator
        if hasattr(underlying, "__cuda_array_interface__"):
            from ._pointer import PointerIterator

            underlying = PointerIterator(underlying)

        self._underlying = underlying
        self._transform_op = make_op_adapter(transform_op)
        self._is_input = is_input
        self._compiled_op = None  # Lazy compiled Op

        # Determine value type
        if value_type is None:
            if is_input:
                # value_type is the return type
                _, value_type = signature_from_annotations(transform_op)
                if value_type is None:
                    value_type = self._transform_op.get_return_type(
                        (underlying.value_type,)
                    )
            else:
                # value_type is the input type
                input_types, _ = signature_from_annotations(transform_op)
                if len(input_types) != 1:
                    raise ValueError(
                        "TransformOutputIterator transform function must take exactly one argument with type annotation"
                    )
                value_type = input_types[0]
        assert value_type is not None
        self._value_type = value_type

        super().__init__(
            state_bytes=bytes(self._underlying.state),
            state_alignment=self._underlying.state_alignment,
            value_type=value_type,
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

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[child_op.ltoir, *child_op.extra_ltoirs],
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
        temp_decl = make_variable_declaration(self._underlying.value_type, "temp")

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* result);
            extern "C" __device__ void {compiled_op.name}(void* input, void* output);

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                {temp_decl}
                {child_op.name}(state, &temp);
                {compiled_op.name}(&temp, result);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[
                compiled_op.ltoir,
                *compiled_op.extra_ltoirs,
                child_op.ltoir,
                *child_op.extra_ltoirs,
            ],
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
        temp_decl = make_variable_declaration(self._underlying.value_type, "temp")

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* value);
            extern "C" __device__ void {compiled_op.name}(void* input, void* output);

            extern "C" __device__ void {symbol}(void* state, void* value) {{
                {temp_decl}
                {compiled_op.name}(value, &temp);
                {child_op.name}(state, &temp);
            }}
        """).strip()

        ltoir = compile_cpp_to_ltoir(source)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[
                compiled_op.ltoir,
                *compiled_op.extra_ltoirs,
                child_op.ltoir,
                *child_op.extra_ltoirs,
            ],
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


class TransformOutputIterator(TransformIterator):
    """
    An iterator that applies a unary function to values before writing them to an underlying iterator.

    Similar to `thrust::transform_output_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1transform__output__iterator.html>`_.

    This is a convenience subclass of TransformIterator configured for output mode.

    Example:
        The code snippet below demonstrates the usage of a ``TransformOutputIterator`` to transform the output
        of a reduction before writing to an output array:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/transform_output_iterator.py
            :language: python
            :start-after: # example-begin

    Args:
        underlying: The underlying iterator or device array
        transform_op: The operation to be applied to values before they are written
        output_value_type: TypeDescriptor for the input value type (optional, will be extracted from annotations if not provided)
    """

    def __init__(self, underlying, transform_op, output_value_type=None):
        super().__init__(underlying, transform_op, output_value_type, is_input=False)

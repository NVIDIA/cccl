# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DiscardIterator implementation."""

from __future__ import annotations

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir
from .._utils.protocols import get_dtype
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..types import TypeDescriptor, from_numpy_dtype
from ._base import IteratorBase
from ._common import CUDA_PREAMBLE


class DiscardIterator(IteratorBase):
    """
    Iterator that discards all reads and writes.
    """

    def __init__(self, reference_iterator=None):
        """
        Create a discard iterator.

        Args:
            reference_iterator: Optional iterator or device array used to infer
                value_type/state_type. Defaults to a temporary byte buffer.
        """
        if reference_iterator is None:
            reference_iterator = TempStorageBuffer(1)

        self._reference_iterator = reference_iterator

        if hasattr(reference_iterator, "__cuda_array_interface__"):
            value_type = from_numpy_dtype(get_dtype(reference_iterator))
            state_bytes = bytes(value_type.dtype.itemsize)
        elif isinstance(reference_iterator, IteratorBase):
            value_type = reference_iterator.value_type
            if isinstance(value_type, TypeDescriptor):
                state_bytes = bytes(value_type.dtype.itemsize)
            else:
                state_bytes = bytes(value_type.info.size)
        else:
            raise TypeError("reference_iterator must be a device array or iterator")

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

    def _make_advance_op(self) -> Op:
        symbol = self._make_advance_symbol()

        source = f"""{CUDA_PREAMBLE}

extern "C" __device__ void {symbol}(void*, void*) {{
}}
"""
        ltoir = compile_cpp_to_ltoir(source)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[],
        )

    def _make_input_deref_op(self) -> Op | None:
        symbol = self._make_input_deref_symbol()

        source = f"""{CUDA_PREAMBLE}

extern "C" __device__ void {symbol}(void*, void*) {{
}}
"""
        ltoir = compile_cpp_to_ltoir(source)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[],
        )

    def _make_output_deref_op(self) -> Op | None:
        symbol = self._make_output_deref_symbol()

        source = f"""{CUDA_PREAMBLE}

extern "C" __device__ void {symbol}(void*, void*) {{
}}
"""
        ltoir = compile_cpp_to_ltoir(source)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[],
        )

    def __add__(self, offset: int) -> "DiscardIterator":
        return DiscardIterator(self._reference_iterator)

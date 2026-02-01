# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PointerIterator implementation - simple bidirectional iterator for device arrays."""

from __future__ import annotations

import ctypes
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir, cpp_type_from_descriptor
from .._utils.protocols import get_data_pointer, get_dtype
from ..types import from_numpy_dtype
from ._base import IteratorBase

CUDA_PREAMBLE = """#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda/std/cstring>
using namespace cuda::std;
"""

if TYPE_CHECKING:
    pass


class PointerIterator(IteratorBase):
    """
    Simple iterator wrapping a device array pointer.

    Supports both input (reading) and output (writing) operations.
    Handles both scalar types (using typed C++ code) and struct types
    (using byte-level memcpy).
    """

    def __init__(self, array):
        """
        Create a pointer iterator from a device array.

        Args:
            array: Device array with __cuda_array_interface__
        """
        # Get pointer and dtype from array
        ptr = get_data_pointer(array)
        dtype = get_dtype(array)
        value_type = from_numpy_dtype(dtype)

        # State is just the pointer
        state_bytes = ctypes.c_void_p(ptr)
        state_bytes_buffer = (ctypes.c_char * 8)()
        ctypes.memmove(state_bytes_buffer, ctypes.byref(state_bytes), 8)
        state_bytes = bytes(state_bytes_buffer)

        self._cpp_type = cpp_type_from_descriptor(value_type)  # None for struct types
        self._element_size = value_type.info.size
        self._array = array  # Keep reference to prevent GC

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # pointer alignment
            value_type=value_type,
        )

    def _make_advance_op(self) -> Op:
        symbol = self._make_advance_symbol()

        if self._cpp_type:
            # Scalar type - use typed pointer arithmetic
            source = dedent(f"""
                {CUDA_PREAMBLE}

                extern "C" __device__ void {symbol}(void* state, void* offset) {{
                    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
                    auto dist = *static_cast<int64_t*>(offset);
                    *ptr_state += dist;
                }}
            """).strip()
        else:
            # Struct type - use byte-level pointer arithmetic
            source = dedent(f"""
                {CUDA_PREAMBLE}

                extern "C" __device__ void {symbol}(void* state, void* offset) {{
                    auto* ptr_state = static_cast<char**>(state);
                    auto dist = *static_cast<int64_t*>(offset);
                    *ptr_state += dist * {self._element_size};
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

        if self._cpp_type:
            # Scalar type - use typed dereference
            source = dedent(f"""
                {CUDA_PREAMBLE}

                extern "C" __device__ void {symbol}(void* state, void* result) {{
                    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
                    *static_cast<{self._cpp_type}*>(result) = **ptr_state;
                }}
            """).strip()
        else:
            # Struct type - use memcpy
            source = dedent(f"""
                {CUDA_PREAMBLE}

                extern "C" __device__ void {symbol}(void* state, void* result) {{
                    auto* ptr_state = static_cast<char**>(state);
                    memcpy(result, *ptr_state, {self._element_size});
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
        symbol = self._make_output_deref_symbol()

        if self._cpp_type:
            # Scalar type - use typed dereference
            source = dedent(f"""
                {CUDA_PREAMBLE}

                extern "C" __device__ void {symbol}(void* state, void* value) {{
                    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
                    **ptr_state = *static_cast<{self._cpp_type}*>(value);
                }}
            """).strip()
        else:
            # Struct type - use memcpy
            source = dedent(f"""
                {CUDA_PREAMBLE}

                extern "C" __device__ void {symbol}(void* state, void* value) {{
                    auto* ptr_state = static_cast<char**>(state);
                    memcpy(*ptr_state, value, {self._element_size});
                }}
            """).strip()

        ltoir = compile_cpp_to_ltoir(source)
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=ltoir,
            extra_ltoirs=[],
        )

    def __add__(self, offset: int):
        dtype = get_dtype(self._array)
        offset_ptr = self._current_pointer() + offset * dtype.itemsize
        return self._clone_with_pointer(offset_ptr)

    def _current_pointer(self) -> int:
        return int.from_bytes(self._state_bytes, sys.byteorder, signed=False)

    def _clone_with_pointer(self, pointer_value: int):
        """Clone this iterator with a different pointer value."""
        clone = PointerIterator.__new__(PointerIterator)
        clone._cpp_type = self._cpp_type
        clone._element_size = self._element_size
        clone._array = self._array
        # Set new pointer value
        state_bytes_buffer = (ctypes.c_char * 8)()
        ptr_obj = ctypes.c_void_p(pointer_value)
        ctypes.memmove(state_bytes_buffer, ctypes.byref(ptr_obj), 8)
        clone._state_bytes = bytes(state_bytes_buffer)
        clone._state_alignment = 8
        clone._value_type = self._value_type
        clone._advance_op = None
        clone._input_deref_op = None
        clone._output_deref_op = None
        clone._uid_cached = None
        return clone

    @property
    def kind(self):
        """
        Return a hashable kind for caching purposes.

        Include _cpp_type and _element_size since they affect generated code.
        Different code paths are taken for scalar vs struct types.
        """
        return (
            type(self).__name__,
            self._value_type,
            self._cpp_type,
            self._element_size,
        )

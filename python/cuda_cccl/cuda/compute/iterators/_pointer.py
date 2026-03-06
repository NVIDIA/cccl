# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PointerIterator implementation - simple bidirectional iterator for device arrays."""

from __future__ import annotations

import ctypes
from textwrap import dedent

from .._bindings import IteratorState, Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir, cpp_type_from_descriptor
from .._utils.protocols import get_data_pointer, get_dtype
from ..types import from_numpy_dtype
from ._base import IteratorBase
from ._common import CUDA_PREAMBLE


class PointerIterator(IteratorBase):
    """
    Simple iterator wrapping a device array pointer.

    Supports both input (reading) and output (writing) operations.
    Handles both scalar types (using typed C++ code) and struct types
    (using byte-level memcpy).

    The data pointer is read lazily from the wrapped array each time
    ``state`` is accessed, so passing a :class:`~cuda.compute.ProxyArray`
    is safe: the null pointer is only materialised at the point where the
    CCCL interop layer actually needs the state bytes (build time), and the
    real pointer is supplied at call time by ``set_cccl_iterator_state``.
    """

    def __init__(self, array):
        """
        Create a pointer iterator from a device array.

        Args:
            array: Device array with ``__cuda_array_interface__``, or a
                :class:`~cuda.compute.ProxyArray` for ahead-of-time
                compilation without real GPU memory.
        """
        dtype = get_dtype(array)
        value_type = from_numpy_dtype(dtype)

        self._cpp_type = cpp_type_from_descriptor(value_type)  # None for struct types
        self._element_size = value_type.info.size
        self._array = array  # Keep reference to prevent GC
        # None means "derive from self._array on demand"; set by _clone_with_pointer
        # when the pointer differs from the array's current address.
        self._override_ptr: int | None = None

        super().__init__(
            state_bytes=b"\x00" * 8,  # placeholder — state property is overridden
            state_alignment=8,  # pointer alignment
            value_type=value_type,
        )

    @property
    def array(self):
        return self._array

    @property
    def state(self) -> IteratorState:
        """Lazily compute pointer state from the wrapped array."""
        ptr = self._current_pointer()
        buf = (ctypes.c_char * 8)()
        ctypes.memmove(buf, ctypes.byref(ctypes.c_void_p(ptr)), 8)
        return IteratorState(bytes(buf))

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
        if self._override_ptr is not None:
            return self._override_ptr
        from .._proxy import ProxyArray

        if isinstance(self._array, ProxyArray):
            return 0  # NULL pointer at build time; real pointer supplied at call time
        return get_data_pointer(self._array)

    def _clone_with_pointer(self, pointer_value: int):
        """Clone this iterator with a different pointer value."""
        clone = PointerIterator(self._array)
        clone._override_ptr = pointer_value
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

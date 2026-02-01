# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CacheModifiedInputIterator implementation."""

from __future__ import annotations

from textwrap import dedent
from typing import Literal

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

# Map modifier names to PTX cache operators and C++ intrinsics
_CACHE_MODIFIERS = {
    "stream": ("cs", "__ldcs"),  # Cache streaming (evict first)
    "global": ("cg", "__ldcg"),  # Cache at L2 only
    "volatile": ("cv", "__ldcv"),  # Don't cache, always fetch
}


class CacheModifiedInputIterator(IteratorBase):
    """
    Iterator that wraps a device pointer with cache-modified loads.

    This iterator uses PTX cache modifiers to control how data is loaded:
    - "stream": Uses streaming loads (ld.global.cs) - hints that data will not be reused
    - "global": Uses global cache loads (ld.global.cg) - caches only at L2
    - "volatile": Uses volatile loads (ld.global.cv) - always fetches from memory

    Supports element types of size 1, 2, 4, 8, or 16 bytes.
    """

    __slots__ = [
        "_modifier",
        "_array",
        "_ptr",
    ]

    def __init__(
        self,
        array,
        modifier: Literal["stream", "global", "volatile"] = "stream",
    ):
        """
        Create a cache-modified input iterator.

        Args:
            array: Device array to wrap (must support __cuda_array_interface__)
            modifier: Cache modifier - "stream", "global", or "volatile"
        """
        if modifier not in _CACHE_MODIFIERS:
            raise ValueError(
                f"Unknown modifier: {modifier}. Must be one of {list(_CACHE_MODIFIERS.keys())}"
            )

        self._modifier = modifier
        self._array = array  # Keep reference to prevent GC
        ptr = get_data_pointer(array)
        dtype = get_dtype(array)

        self._ptr = ptr
        value_type = from_numpy_dtype(dtype)

        # Cache-modified loads only supported for power-of-two sizes up to 16 bytes
        # These correspond to PTX instructions: ld.global.{modifier}.b{8,16,32,64,128}
        if value_type.size not in (1, 2, 4, 8, 16):
            raise ValueError(
                f"CacheModifiedInputIterator only supports types of size 1, 2, 4, 8, or 16 bytes. "
                f"Got type with size {value_type.size} bytes. "
                f"This matches PTX cache-modified load instruction limitations."
            )

        # State is just the pointer (8 bytes on 64-bit)
        import struct

        state_bytes = struct.pack("Q", ptr)

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # Pointer alignment
            value_type=value_type,
        )

    def _make_advance_op(self) -> Op:
        symbol = self._make_advance_symbol()
        cpp_type = cpp_type_from_descriptor(self._value_type)

        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                auto* s = static_cast<{cpp_type}**>(state);
                auto dist = *static_cast<uint64_t*>(offset);
                *s += dist;
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
        _, intrinsic = _CACHE_MODIFIERS[self._modifier]

        # Use cache-modified intrinsic for all supported sizes (1, 2, 4, 8, 16 bytes)
        # These correspond to PTX instructions: ld.global.{modifier}.b{8,16,32,64,128}
        # Note: __ldcs, __ldcg, __ldcv intrinsics work for all these sizes
        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                auto* ptr = *static_cast<{cpp_type}**>(state);
                *static_cast<{cpp_type}*>(result) = {intrinsic}(ptr);
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
        # Cache-modified iterator is input-only
        return None

    def __add__(self, offset: int) -> "CacheModifiedInputIterator":
        """Advance the iterator by offset elements."""
        import struct

        from .._utils.protocols import get_dtype

        dtype = get_dtype(self._array)
        offset_ptr = self._ptr + offset * dtype.itemsize

        # Create a new instance with the offset pointer
        clone = CacheModifiedInputIterator.__new__(CacheModifiedInputIterator)
        clone._modifier = self._modifier
        clone._array = self._array
        clone._ptr = offset_ptr
        clone._state_bytes = struct.pack("Q", offset_ptr)
        clone._state_alignment = 8
        clone._value_type = self._value_type
        clone._uid_cached = None
        return clone

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return (
            "CacheModifiedInputIterator",
            self._modifier,
            self._value_type,
        )

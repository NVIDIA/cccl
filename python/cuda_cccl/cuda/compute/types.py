# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Standalone type descriptors for cuda.compute.

This module provides type descriptors that can be used to specify argument
and return types for CompiledOp and CompiledIterator without depending on Numba.

Each type descriptor maps directly to the underlying CCCL TypeInfo structure
with size, alignment, and type enumeration.

Example:
    from cuda.compute import types, CompiledOp

    add_op = CompiledOp(
        ltoir=ltoir_bytes,
        name="my_add",
        arg_types=(types.int32, types.int32),
        return_type=types.int32,
    )
"""

import numpy as np

from ._bindings import TypeEnum, TypeInfo

# Mapping from numpy dtype to TypeEnum
_NUMPY_DTYPE_TO_ENUM = {
    np.dtype("int8"): TypeEnum.INT8,
    np.dtype("int16"): TypeEnum.INT16,
    np.dtype("int32"): TypeEnum.INT32,
    np.dtype("int64"): TypeEnum.INT64,
    np.dtype("uint8"): TypeEnum.UINT8,
    np.dtype("uint16"): TypeEnum.UINT16,
    np.dtype("uint32"): TypeEnum.UINT32,
    np.dtype("uint64"): TypeEnum.UINT64,
    np.dtype("float16"): TypeEnum.FLOAT16,
    np.dtype("float32"): TypeEnum.FLOAT32,
    np.dtype("float64"): TypeEnum.FLOAT64,
}


class _TypeDescriptor:
    """
    A type descriptor that wraps size, alignment, and type enumeration.

    This is used to specify types for pre-compiled operations without
    requiring Numba's type system.
    """

    __slots__ = ("size", "alignment", "_type_enum", "_name")

    def __init__(self, size: int, alignment: int, type_enum: TypeEnum, name: str):
        self.size = size
        self.alignment = alignment
        self._type_enum = type_enum
        self._name = name

    def to_type_info(self) -> TypeInfo:
        """Convert this type descriptor to a CCCL TypeInfo object."""
        return TypeInfo(self.size, self.alignment, self._type_enum)

    def __repr__(self) -> str:
        return f"types.{self._name}"

    def __eq__(self, other) -> bool:
        if isinstance(other, _TypeDescriptor):
            return (
                self.size == other.size
                and self.alignment == other.alignment
                and self._type_enum == other._type_enum
            )
        return False

    def __hash__(self) -> int:
        return hash((self.size, self.alignment, self._type_enum))


# Signed integer types
int8 = _TypeDescriptor(1, 1, TypeEnum.INT8, "int8")
int16 = _TypeDescriptor(2, 2, TypeEnum.INT16, "int16")
int32 = _TypeDescriptor(4, 4, TypeEnum.INT32, "int32")
int64 = _TypeDescriptor(8, 8, TypeEnum.INT64, "int64")

# Unsigned integer types
uint8 = _TypeDescriptor(1, 1, TypeEnum.UINT8, "uint8")
uint16 = _TypeDescriptor(2, 2, TypeEnum.UINT16, "uint16")
uint32 = _TypeDescriptor(4, 4, TypeEnum.UINT32, "uint32")
uint64 = _TypeDescriptor(8, 8, TypeEnum.UINT64, "uint64")

# Floating point types
float16 = _TypeDescriptor(2, 2, TypeEnum.FLOAT16, "float16")
float32 = _TypeDescriptor(4, 4, TypeEnum.FLOAT32, "float32")
float64 = _TypeDescriptor(8, 8, TypeEnum.FLOAT64, "float64")


def custom_type(size: int, alignment: int) -> _TypeDescriptor:
    """
    Create a custom type descriptor for user-defined struct types.

    This is useful for pre-compiled operations that work with custom
    struct types where the exact layout is known.

    Args:
        size: Size of the type in bytes
        alignment: Alignment of the type in bytes

    Returns:
        A _TypeDescriptor for the custom type

    Example:
        # A custom struct with 16 bytes and 8-byte alignment
        my_struct_type = types.custom_type(16, 8)
    """
    return _TypeDescriptor(
        size, alignment, TypeEnum.STORAGE, f"custom({size}, {alignment})"
    )


def from_numpy_dtype(dtype: np.dtype) -> _TypeDescriptor:
    """
    Create a type descriptor from a numpy dtype.

    This handles both scalar dtypes (int32, float64, etc.) and structured
    dtypes (record types). For structured dtypes, a custom type with the
    appropriate size and alignment is returned.

    Args:
        dtype: A numpy dtype

    Returns:
        A _TypeDescriptor for the dtype

    Example:
        from cuda.compute.types import from_numpy_dtype
        import numpy as np

        int_type = from_numpy_dtype(np.dtype('int32'))
        struct_type = from_numpy_dtype(np.dtype([('x', 'i4'), ('y', 'f8')]))
    """
    dtype = np.dtype(dtype)  # Ensure it's a dtype object
    if dtype in _NUMPY_DTYPE_TO_ENUM:
        return _TypeDescriptor(
            dtype.itemsize, dtype.alignment, _NUMPY_DTYPE_TO_ENUM[dtype], dtype.name
        )

    # Handle complex types (kind 'c' = complex floating)
    if dtype.kind == "c":
        return _TypeDescriptor(
            dtype.itemsize, dtype.alignment, TypeEnum.STORAGE, dtype.name
        )

    # For structured/record types, use STORAGE enum
    return _TypeDescriptor(
        dtype.itemsize, dtype.alignment, TypeEnum.STORAGE, f"struct({dtype})"
    )


__all__ = [
    # Signed integers
    "int8",
    "int16",
    "int32",
    "int64",
    # Unsigned integers
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    # Floating point
    "float16",
    "float32",
    "float64",
    # Custom types
    "custom_type",
    # Conversion
    "from_numpy_dtype",
]

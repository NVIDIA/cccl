# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import numpy as np

from ._bindings import TypeEnum, TypeInfo

_ENUM_TO_DTYPE: dict[TypeEnum, np.dtype] = {
    TypeEnum.INT8: np.dtype("int8"),
    TypeEnum.INT16: np.dtype("int16"),
    TypeEnum.INT32: np.dtype("int32"),
    TypeEnum.INT64: np.dtype("int64"),
    TypeEnum.UINT8: np.dtype("uint8"),
    TypeEnum.UINT16: np.dtype("uint16"),
    TypeEnum.UINT32: np.dtype("uint32"),
    TypeEnum.UINT64: np.dtype("uint64"),
    TypeEnum.FLOAT16: np.dtype("float16"),
    TypeEnum.FLOAT32: np.dtype("float32"),
    TypeEnum.FLOAT64: np.dtype("float64"),
    TypeEnum.BOOLEAN: np.dtype("bool"),
}


class TypeDescriptor:
    def __init__(self, size: int, alignment: int, type_enum: TypeEnum):
        self._type_info = TypeInfo(size, alignment, type_enum)
        self._dtype = _ENUM_TO_DTYPE.get(type_enum, np.dtype(f"V{size}"))

    @property
    def info(self) -> TypeInfo:
        """Return the TypeInfo for this type."""
        return self._type_info

    @property
    def size(self):
        return self._type_info.size

    @property
    def alignment(self):
        return self._type_info.alignment

    @property
    def dtype(self) -> np.dtype:
        """Return the numpy dtype for this type."""
        return self._dtype

    def pointer(self) -> "PointerTypeDescriptor":
        """Create a pointer type to this type."""
        return PointerTypeDescriptor(self)

    def __repr__(self) -> str:
        return f"TypeDescriptor({self._dtype})"

    def __eq__(self, other):
        if not isinstance(other, TypeDescriptor):
            return False
        return self._dtype == other._dtype

    def __hash__(self):
        return hash(self._dtype)


class StructTypeDescriptor(TypeDescriptor):
    def __init__(
        self,
        fields: dict[str, "TypeDescriptor"],
        name: str = "AnonStruct",
    ):
        dtype = _build_struct_dtype(fields)
        if not dtype.isalignedstruct:
            raise ValueError(f"dtype {dtype} must be aligned")
        # Structs use STORAGE type enum
        super().__init__(dtype.itemsize, dtype.alignment, TypeEnum.STORAGE)
        self._dtype = dtype
        self._fields = fields
        self._name = name

    def __repr__(self) -> str:
        return f"StructTypeDescriptor({self._name}, dtype={self._dtype})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def fields(self) -> dict[str, "TypeDescriptor"]:
        return self._fields

    def layout_key(self) -> tuple[tuple[str, "TypeDescriptor"], ...]:
        """Return a stable, hashable key for this struct layout."""
        return tuple(self._fields.items())

    def __eq__(self, other):
        if not isinstance(other, StructTypeDescriptor):
            return False
        # Compare by fields (TypeDescriptors) to correctly distinguish structs
        # with different pointer pointee types, which have identical numpy dtypes
        # (both uint64) but different semantics.
        return self._fields == other._fields

    def __hash__(self):
        # Convert to a hashable tuple of (name, type_descriptor) pairs
        return hash(tuple(sorted((k, v) for k, v in self._fields.items())))


class PointerTypeDescriptor(TypeDescriptor):
    def __init__(self, pointee: TypeDescriptor):
        # Pointer is 8 bytes, 8-byte aligned, represented as uint64
        super().__init__(8, 8, TypeEnum.UINT64)
        self._pointee = pointee

    @property
    def pointee(self) -> TypeDescriptor:
        """Return the type this pointer points to."""
        return self._pointee

    def __repr__(self) -> str:
        return f"PointerTypeDescriptor({self._pointee})"

    def __eq__(self, other):
        if not isinstance(other, PointerTypeDescriptor):
            return False
        return self._pointee == other._pointee

    def __hash__(self):
        return hash(("PointerTypeDescriptor", self._pointee))


def _build_struct_dtype(
    fields: dict[str, TypeDescriptor],
) -> np.dtype:
    dtype_list = []
    for field_name, field_type in fields.items():
        dtype_list.append((field_name, field_type.dtype))
    return np.dtype(dtype_list, align=True)


def struct(
    fields: dict[str, TypeDescriptor],
    name: str = "AnonStruct",
) -> StructTypeDescriptor:
    """Create a type descriptor for a struct."""
    return StructTypeDescriptor(fields, name=name)


def pointer(pointee: TypeDescriptor) -> PointerTypeDescriptor:
    """
    Create a pointer to the given type.
    """
    return PointerTypeDescriptor(pointee)


def from_numpy_dtype(dtype: np.dtype | type) -> TypeDescriptor:
    """
    Convert a numpy dtype (or numpy type) to a TypeDescriptor.

    Handles POD types and structured dtypes (recursively for nested structs).
    """
    dtype = np.dtype(dtype)

    # Check if it's a known POD type
    td = _DTYPE_TO_TD.get(dtype)
    if td is not None:
        return td

    # Handle structured dtypes (structs)
    if dtype.names is not None:
        fields: dict[str, TypeDescriptor] = {}
        assert dtype.fields is not None
        for name in dtype.names:
            field_info = dtype.fields[name]
            field_dtype = field_info[0]
            fields[name] = from_numpy_dtype(field_dtype)
        return struct(fields)  # type: ignore[arg-type]

    # Some other NumPy type (e.g., complex64) for which we don't
    # have a specific TypeDescriptor. Use STORAGE and preserve the original dtype.
    td = TypeDescriptor(dtype.itemsize, dtype.alignment, TypeEnum.STORAGE)
    td._dtype = dtype
    return td


# Signed integer types
int8 = TypeDescriptor(1, 1, TypeEnum.INT8)
int16 = TypeDescriptor(2, 2, TypeEnum.INT16)
int32 = TypeDescriptor(4, 4, TypeEnum.INT32)
int64 = TypeDescriptor(8, 8, TypeEnum.INT64)

# Unsigned integer types
uint8 = TypeDescriptor(1, 1, TypeEnum.UINT8)
uint16 = TypeDescriptor(2, 2, TypeEnum.UINT16)
uint32 = TypeDescriptor(4, 4, TypeEnum.UINT32)
uint64 = TypeDescriptor(8, 8, TypeEnum.UINT64)

# Floating point types
float16 = TypeDescriptor(2, 2, TypeEnum.FLOAT16)
float32 = TypeDescriptor(4, 4, TypeEnum.FLOAT32)
float64 = TypeDescriptor(8, 8, TypeEnum.FLOAT64)

# Boolean
boolean = TypeDescriptor(1, 1, TypeEnum.BOOLEAN)


# Mapping from numpy dtype to TypeDescriptor for POD types
_DTYPE_TO_TD: dict[np.dtype, TypeDescriptor] = {
    np.dtype("int8"): int8,
    np.dtype("int16"): int16,
    np.dtype("int32"): int32,
    np.dtype("int64"): int64,
    np.dtype("uint8"): uint8,
    np.dtype("uint16"): uint16,
    np.dtype("uint32"): uint32,
    np.dtype("uint64"): uint64,
    np.dtype("float16"): float16,
    np.dtype("float32"): float32,
    np.dtype("float64"): float64,
    np.dtype("bool"): boolean,
}


def to_ctypes_type(td: TypeDescriptor):
    """Convert a TypeDescriptor to a ctypes type."""
    return np.ctypeslib.as_ctypes_type(td.dtype)


__all__ = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "boolean",
    "struct",
    "pointer",
    "from_numpy_dtype",
    "to_ctypes_type",
]

# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This module provides `gpu_struct`, a factory for producing struct types.

"""

import functools
from types import new_class
from typing import Any, ClassVar, TypeGuard, Union, cast, get_type_hints

import numpy as np

from . import types


def gpu_struct(
    field_dict: Union[dict, np.dtype, type],
    name: str = "AnonymousStruct",
):
    """
    A factory for creating struct types.

    Args:
        field_dict
                A dictionary, numpy dtype, or annotated class providing the
            mapping of field names to data types.

        name
            The name of the struct type that will be returned.

    Returns:
        A struct class helpful for writing operations on struct values.
    """

    # Handle numpy dtype input
    if isinstance(field_dict, np.dtype):
        if field_dict.type != np.void or field_dict.fields is None:
            field_dict = {}
        else:
            field_dict = {
                name: field_info[0] for name, field_info in field_dict.fields.items()
            }

    # Handle annotated class (decorator usage)
    if isinstance(field_dict, type) and hasattr(field_dict, "__annotations__"):
        name = field_dict.__name__
        field_dict = get_type_hints(field_dict)

    # At this point, field_dict must be a dict
    assert isinstance(field_dict, dict)

    # Normalize fields for storage on the struct class
    field_spec = {}
    for key, val in field_dict.items():
        if _is_struct_type(val):
            field_spec[key] = val
        elif isinstance(val, dict):
            # Nested struct definition - recursively create inner struct
            field_spec[key] = gpu_struct(val, name=key)
        else:
            field_spec[key] = val

    # Create a simple Python class for user-facing struct values
    struct_class = cast(type[_Struct], new_class(name, bases=(_Struct,)))
    struct_class._field_spec = field_spec
    struct_class._type_descriptor = _get_struct_type_descriptor(struct_class)  # type: ignore[arg-type]
    struct_class.dtype = _get_struct_record_dtype(struct_class)  # type: ignore[arg-type]

    return struct_class


class _Struct:
    """Internal base class for all gpu_structs."""

    _field_spec: ClassVar[dict[str, Any]]
    _type_descriptor: ClassVar[types.StructTypeDescriptor]
    dtype: ClassVar[np.dtype]
    _fields: dict[str, Any]

    @classmethod
    def _fields_from_args(cls, *args, **kwargs):
        field_spec = cls._field_spec

        if args and isinstance(args[0], dict):
            fields = args[0]
        elif args:
            assert len(args) == len(field_spec), (
                f"Expected {len(field_spec)} arguments, got {len(args)}"
            )
            fields = dict(zip(field_spec.keys(), args))
        else:
            fields = kwargs

        assert fields.keys() == field_spec.keys()

        return {
            name: _coerce_value(field_spec[name], fields[name]) for name in field_spec
        }

    def __init__(self, *args, **kwargs):
        """Supporting construction from positional, keyword, and dict arguments."""

        self._fields = self._fields_from_args(*args, **kwargs)

        for name, value in self._fields.items():
            setattr(self, name, value)

        # NumPy array representation:
        self._data = np.asarray(_as_numpy_record_value(self))
        self.__array_interface__ = self._data.__array_interface__


def _as_numpy_record_value(val) -> np.void:
    """Convert a gpu_struct *value* to a numpy record."""

    def _fields_to_tuples(fields_dict: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            _fields_to_tuples(v._fields) if isinstance(v, _Struct) else v
            for v in fields_dict.values()
        )

    return np.void(
        _fields_to_tuples(val._fields),
        dtype=_get_struct_record_dtype(type(val)),  # type: ignore[arg-type]
    )


@functools.cache
def _get_struct_record_dtype(struct_class: type) -> np.dtype:
    return _get_struct_type_descriptor(struct_class).dtype


def _coerce_value(field_type, value: Any) -> Any:
    if isinstance(value, _Struct):
        return value

    if isinstance(field_type, np.dtype):
        return field_type.type(value)

    if isinstance(field_type, type) and issubclass(field_type, np.generic):
        return field_type(value)

    if isinstance(value, tuple):
        return field_type(*value)

    if isinstance(value, dict):
        return field_type(**value)

    # field_type is a class (e.g., another gpu_struct)
    raise TypeError(f"Cannot coerce {type(value).__name__} into {field_type.__name__}")


def _is_struct_type(typ: Any) -> TypeGuard[type[_Struct]]:
    """Check if a type is a GPU struct class."""
    return isinstance(typ, type) and issubclass(typ, _Struct)


@functools.cache
def _get_struct_type_descriptor(
    struct_class: type,
) -> types.StructTypeDescriptor:
    type_descriptors = _field_spec_to_type_descriptors(
        struct_class._field_spec  # type: ignore[attr-defined]
    )
    return types.struct(type_descriptors, name=struct_class.__name__)


def _field_spec_to_type_descriptors(
    field_spec: dict[str, Any],
) -> dict[str, types.TypeDescriptor]:
    type_descriptors = {}
    for key, val in field_spec.items():
        if isinstance(val, types.TypeDescriptor):
            type_descriptors[key] = val
        elif _is_struct_type(val):
            type_descriptors[key] = val._type_descriptor
        elif isinstance(val, np.dtype):
            type_descriptors[key] = types.from_numpy_dtype(val)
        else:
            type_descriptors[key] = types.from_numpy_dtype(np.dtype(val))
    return type_descriptors

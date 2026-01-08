# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
GPU struct types for cuda.compute.

This module provides `gpu_struct`, a factory for creating struct types that can be
used with both JIT-compiled (Numba) and pre-compiled (BYOC) operations.

The struct types have lazy Numba registration - Numba is only imported and the
type is only registered when actually needed for JIT compilation.

Example:
    >>> from cuda.compute import gpu_struct
    >>> import numpy as np
    >>> S = gpu_struct({'x': np.int32, 'y': np.int64})
    >>> s = S(x=1, y=2)
"""

from typing import Any, Union

import numpy as np

from .types import TypeEnum, _TypeDescriptor


class _Struct:
    """
    Base class for all gpu_struct types.

    Provides:
    - Field storage and access
    - A `_type_descriptor` for use with CompiledOp/CompiledIterator
    - Lazy Numba registration via `_numba_type` property
    """

    _fields: dict[str, Any]
    # Class attribute: field name -> numpy dtype or nested struct
    _field_spec: dict[str, Any]
    _dtype: np.dtype  # Class attribute: numpy record dtype
    _numba_registered: bool = False  # Class attribute: whether registered with Numba

    @classmethod
    def _get_type_descriptor(cls) -> _TypeDescriptor:
        """Get a TypeDescriptor for this struct type."""
        dtype = cls._dtype
        return _TypeDescriptor(
            dtype.itemsize, dtype.alignment, TypeEnum.STORAGE, f"struct({cls.__name__})"
        )

    @classmethod
    def _get_numba_type(cls):
        """
        Get the Numba type for this struct, registering it if necessary.

        This is called lazily by _JitOp.compile() when the struct is used
        with a JIT-compiled operation.
        """
        if not cls._numba_registered:
            from ._numba.struct import _register_struct_with_numba

            _register_struct_with_numba(cls)
            cls._numba_registered = True

        from numba.core.extending import as_numba_type

        return as_numba_type(cls)


def _normalize_field_spec(
    field_dict: Union[dict, np.dtype, type],
) -> tuple[str, dict[str, Any]]:
    """
    Normalize input to gpu_struct into a name and field spec dictionary.

    Returns:
        Tuple of (struct_name, field_spec_dict)
    """
    name = "AnonymousStruct"

    # Handle numpy dtype input
    if isinstance(field_dict, np.dtype):
        if field_dict.fields is None:
            raise ValueError("Expected a structured numpy dtype")
        field_dict = {
            fname: field_info[0] for fname, field_info in field_dict.fields.items()
        }

    # Handle annotated class (decorator usage)
    if isinstance(field_dict, type):
        if hasattr(field_dict, "__annotations__"):
            name = field_dict.__name__
            field_dict = field_dict.__annotations__
        else:
            raise ValueError("Expected an annotated class")

    return name, field_dict


def _compute_numpy_dtype(field_spec: dict[str, Any]) -> np.dtype:
    """Compute the numpy record dtype for a field spec."""

    def _field_to_dtype(field_type):
        if isinstance(field_type, type) and issubclass(field_type, _Struct):
            return field_type._dtype
        if isinstance(field_type, np.dtype):
            return field_type
        return np.dtype(field_type)

    return np.dtype(
        [(name, _field_to_dtype(ftype)) for name, ftype in field_spec.items()],
        align=True,
    )


def _coerce_value(field_type, value: Any) -> Any:
    """Coerce a value to the appropriate type for a field."""
    if isinstance(value, _Struct):
        return value
    if isinstance(field_type, type) and issubclass(field_type, _Struct):
        if isinstance(value, tuple):
            return field_type(*value)
        if isinstance(value, dict):
            return field_type(**value)
    return value


def _fields_to_numpy(fields: dict[str, Any], dtype: np.dtype) -> np.void:
    """Convert struct fields to a numpy record value."""

    def _to_tuple(fields_dict: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            _to_tuple(v._fields) if isinstance(v, _Struct) else v
            for v in fields_dict.values()
        )

    return np.void(_to_tuple(fields), dtype=dtype)


def gpu_struct(
    field_dict: Union[dict, np.dtype, type], name: str = "AnonymousStruct"
) -> type:
    """
    Factory for creating struct types with pass-by-value semantics.

    Creates a struct type that can be used with both JIT-compiled (Numba)
    and pre-compiled (BYOC) operations. Numba registration is deferred
    until the struct is actually used with a JIT-compiled operation.

    Args:
        field_dict:
            A dictionary, numpy dtype, or annotated class providing the
            mapping of field names to data types.
        name:
            The name of the struct type that will be returned.

    Returns:
        A Python class representing the struct type.

    Examples:

    Construction from a dictionary:

    .. code-block:: python

        S = gpu_struct({'x': np.int32, 'y': np.int64})

    Construction from a numpy dtype:

    .. code-block:: python

        S = gpu_struct(np.dtype([('x', 'i4'), ('y', 'i8')]))

    Construction from an annotated class:

    .. code-block:: python

        @gpu_struct
        class MyStruct:
            x: np.int32
            y: np.int64
    """
    # Normalize input
    inferred_name, field_spec = _normalize_field_spec(field_dict)
    if name == "AnonymousStruct" and inferred_name != "AnonymousStruct":
        name = inferred_name

    # Recursively convert nested dicts to gpu_structs
    processed_spec = {}
    for fname, ftype in field_spec.items():
        if isinstance(ftype, dict):
            processed_spec[fname] = gpu_struct(ftype, f"{name}_{fname}")
        elif isinstance(ftype, np.dtype) and ftype.fields is not None:
            processed_spec[fname] = gpu_struct(ftype, f"{name}_{fname}")
        else:
            processed_spec[fname] = ftype

    # Compute numpy dtype
    dtype = _compute_numpy_dtype(processed_spec)

    # Create the struct class
    class StructClass(_Struct):
        pass

    StructClass.__name__ = name
    StructClass.__qualname__ = name
    StructClass._field_spec = processed_spec
    StructClass._dtype = dtype
    StructClass._numba_registered = False

    # Add __init__ method
    def __init__(self, *args, **kwargs):
        field_spec = self.__class__._field_spec

        if args and isinstance(args[0], dict):
            fields = args[0]
        elif args:
            if len(args) != len(field_spec):
                raise ValueError(
                    f"Expected {len(field_spec)} arguments, got {len(args)}"
                )
            fields = dict(zip(field_spec.keys(), args))
        else:
            fields = kwargs

        if set(fields.keys()) != set(field_spec.keys()):
            raise ValueError(
                f"Expected fields {set(field_spec.keys())}, got {set(fields.keys())}"
            )

        # Coerce values
        self._fields = {
            fname: _coerce_value(field_spec[fname], fields[fname])
            for fname in field_spec
        }

        # Set attributes
        for fname, value in self._fields.items():
            setattr(self, fname, value)

        # NumPy array representation
        self._data = np.asarray(_fields_to_numpy(self._fields, self.__class__._dtype))
        self.__array_interface__ = self._data.__array_interface__

    StructClass.__init__ = __init__

    # Add dtype property
    StructClass.dtype = dtype

    # Register with Numba if available (for backward compatibility)
    try:
        from ._numba.struct import _register_struct_with_numba

        _register_struct_with_numba(StructClass)
        StructClass._numba_registered = True
    except ImportError:
        pass  # Numba not installed, skip registration

    return StructClass


def make_struct_type(name, field_names, field_types):
    """
    Create a struct type from field names and types.

    This is a lower-level API. Prefer using gpu_struct() instead.
    """
    field_dict = dict(zip(field_names, field_types))
    return gpu_struct(field_dict, name)

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
GPU struct types with pass-by-value semantics for Numba CUDA device functions.

This module provides `gpu_struct`, a factory for producing "struct" types that have
pass-by-value semantics when used with Numba CUDA device functions.

Note: This module requires Numba. The implementation is in _numba/struct.py.

Example:
    >>> from cuda.compute import gpu_struct
    >>> S = gpu_struct({'x': np.int32, 'y': np.int64})
"""

from typing import Any, Union

import numpy as np


def gpu_struct(field_dict: Union[dict, np.dtype, type], name: str = "AnonymousStruct"):
    """
    A factory for creating struct types with pass-by-value semantics in Numba.

    Note: This function requires Numba to be installed.

    Args:
        field_dict
            A dictionary, numpy dtype, or annotated class providing the
            mapping of field names to data types.

        name
            The name of the struct type that will be returned.

    Returns:
        A Python class that has been registered with Numba as a struct type.
        ``as_numba_type()`` can be used to get the underlying Numba type.
        Instances of this class can be passed as arguments to device functions.

    Examples:

    Construction from a dictionary.

    .. code-block:: python

        S = gpu_struct({'x': np.int32, 'y': np.int64})

    Construction from a numpy dtype.

    .. code-block:: python

        S = gpu_struct(np.dtype([('x', 'i4'), ('y', 'i8')]))

    Construction from an annotated class.

    .. code-block:: python

        @gpu_struct
        class MyStruct:
            x: np.int32
            y: np.int64
    """
    from ._numba.struct import gpu_struct as _gpu_struct

    return _gpu_struct(field_dict, name)


def make_struct_type(name, field_names, field_types):
    """
    Core factory function for creating struct types with pass-by-value semantics.

    Note: This function requires Numba to be installed.
    """
    from ._numba.struct import make_struct_type as _make_struct_type

    return _make_struct_type(name, field_names, field_types)


# Re-export _Struct for isinstance checks
class _Struct:
    """Internal base class for all gpu_structs."""

    _fields: dict[str, Any]


# Update _Struct to point to the actual implementation
def __getattr__(name):
    if name == "_Struct":
        from ._numba.struct import _Struct

        return _Struct
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

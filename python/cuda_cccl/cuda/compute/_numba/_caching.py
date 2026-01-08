# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Caching utilities for Numba-based operations.

This module provides utilities for hashing and comparing functions
based on their bytecode, constants, and closures for caching purposes.
"""


def _hash_device_array_like(value):
    # hash based on pointer, shape, and dtype
    ptr = value.__cuda_array_interface__["data"][0]
    shape = value.__cuda_array_interface__["shape"]
    dtype = value.__cuda_array_interface__["typestr"]
    return hash((ptr, shape, dtype))


def _make_hashable(value):
    import numba.cuda.dispatcher

    from ..typing import DeviceArrayLike

    if isinstance(value, numba.cuda.dispatcher.CUDADispatcher):
        return CachableFunction(value.py_func)
    elif isinstance(value, DeviceArrayLike):
        return _hash_device_array_like(value)
    elif isinstance(value, (list, tuple)):
        return tuple(_make_hashable(v) for v in value)
    elif isinstance(value, dict):
        return tuple(
            sorted((_make_hashable(k), _make_hashable(v)) for k, v in value.items())
        )
    else:
        return id(value)


class CachableFunction:
    """
    A type that wraps a function and provides custom comparison
    (__eq__) and hash (__hash__) implementations.

    The purpose of this class is to enable caching and comparison of
    functions based on their bytecode, constants, and closures, while
    ignoring other attributes such as their names or docstrings.
    """

    def __init__(self, func):
        self._func = func

        closure = func.__closure__ if func.__closure__ is not None else []
        contents = []
        # if any of the contents is a numba.cuda.dispatcher.CUDADispatcher
        # use the function for caching purposes:
        for cell in closure:
            contents.append(_make_hashable(cell.cell_contents))
        self._identity = (
            func.__name__,
            func.__code__.co_code,
            func.__code__.co_consts,
            tuple(contents),
            tuple(
                _make_hashable(func.__globals__.get(name, None))
                for name in func.__code__.co_names
            ),
        )

    def __eq__(self, other):
        return self._identity == other._identity

    def __hash__(self):
        return hash(self._identity)

    def __repr__(self):
        return str(self._func)

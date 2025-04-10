# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

import cupy as cp


def cache_with_key(key):
    """
    Decorator to cache the result of the decorated function.  Uses the
    provided `key` function to compute the key for cache lookup. `key`
    receives all arguments passed to the function.

    Notes
    -----
    The CUDA compute capability of the current device is appended to
    the cache key returned by `key`.
    """

    def deco(func):
        cache = {}

        @functools.wraps(func)
        def inner(*args, **kwargs):
            cc = cp.cuda.Device().compute_capability
            cache_key = (key(*args, **kwargs), cc)
            if cache_key not in cache:
                result = func(*args, **kwargs)
                cache[cache_key] = result
            return cache[cache_key]

        def cache_clear():
            cache.clear()

        inner.cache_clear = cache_clear
        return inner

    return deco


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
        self._identity = (
            func.__code__.co_code,
            func.__code__.co_consts,
            tuple(cell.cell_contents for cell in closure),
        )

    def __eq__(self, other):
        return self._identity == other._identity

    def __hash__(self):
        return hash(self._identity)

    def __repr__(self):
        return str(self._func)

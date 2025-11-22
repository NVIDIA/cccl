# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

from cuda.core.experimental import Device


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
            cc = Device().compute_capability
            cache_key = (key(*args, **kwargs), tuple(cc))
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
        # if the function is a `@cuda.jit'd` function, use the wrapped
        # function for caching purposes.
        import numba.cuda.dispatcher

        if isinstance(func, numba.cuda.dispatcher.CUDADispatcher):
            func = func.py_func

        self._func = func

        self._identity = (
            func.__code__.co_code,
            func.__code__.co_consts,
            tuple(func.__closure__ if func.__closure__ is not None else []),
            tuple(func.__globals__.get(name, None) for name in func.__code__.co_names),
        )

    def __eq__(self, other):
        return self._identity == other._identity

    def __hash__(self):
        return hash(self._identity)

    def __repr__(self):
        return str(self._func)

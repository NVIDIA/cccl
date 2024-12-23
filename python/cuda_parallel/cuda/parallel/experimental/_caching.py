# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from numba import cuda


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
            cc = cuda.get_current_device().compute_capability
            cache_key = (key(*args, **kwargs), *cc)
            if cache_key not in cache:
                result = func(*args, **kwargs)
                cache[cache_key] = result
            return cache[cache_key]

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

    def __eq__(self, other):
        func1, func2 = self._func, other._func

        # return True if the functions compare equal for
        # caching purposes, False otherwise
        code1 = func1.__code__
        code2 = func2.__code__

        return (
            code1.co_code == code2.co_code
            and code1.co_consts == code2.co_consts
            and func1.__closure__ == func2.__closure__
        )

    def __hash__(self):
        return hash(
            (
                self._func.__code__.co_code,
                self._func.__code__.co_consts,
                self._func.__closure__,
            )
        )

    def __repr__(self):
        return str(self._func)

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

from cuda.core.experimental import Device

# Central registry of all algorithm caches
_cache_registry: dict[str, object] = {}


def cache_with_key(key):
    """
    Decorator to cache the result of the decorated function.  Uses the
    provided `key` function to compute the key for cache lookup. `key`
    receives all arguments passed to the function.

    Notes
    -----
    The CUDA compute capability of the current device is appended to
    the cache key returned by `key`.

    The decorated function is automatically registered in the central
    cache registry for easy cache management.
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

        # Register the cache in the central registry
        cache_name = func.__qualname__
        _cache_registry[cache_name] = inner

        return inner

    return deco


def clear_all_caches():
    """
    Clear all algorithm caches.

    This function clears all cached algorithm build results, forcing
    recompilation on the next invocation. Useful for benchmarking
    compilation time.

    Example
    -------
    >>> import cuda.compute
    >>> cuda.compute.clear_all_caches()
    """
    for cached_func in _cache_registry.values():
        cached_func.cache_clear()


class CachableFunction:
    """
    A type that wraps a function and provides custom comparison
    (__eq__) and hash (__hash__) implementations.

    The purpose of this class is to enable caching and comparison of
    functions based on their bytecode, constants, and closures, while
    ignoring other attributes such as their names or docstrings.
    """

    def __init__(self, func):
        import numba.cuda.dispatcher

        self._func = func

        closure = func.__closure__ if func.__closure__ is not None else []
        contents = []
        # if any of the contents is a numba.cuda.dispatcher.CUDADispatcher
        # use the function for caching purposes:
        for cell in closure:
            if isinstance(cell.cell_contents, numba.cuda.dispatcher.CUDADispatcher):
                contents.append(CachableFunction(cell.cell_contents.py_func))
            else:
                contents.append(cell.cell_contents)
        self._identity = (
            func.__name__,
            func.__code__.co_code,
            func.__code__.co_consts,
            tuple(contents),
            tuple(func.__globals__.get(name, None) for name in func.__code__.co_names),
        )

    def __eq__(self, other):
        return self._identity == other._identity

    def __hash__(self):
        return hash(self._identity)

    def __repr__(self):
        return str(self._func)

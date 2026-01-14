# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

try:
    from cuda.core import Device
except ImportError:
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

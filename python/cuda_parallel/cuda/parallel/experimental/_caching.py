# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools


def cache_with_key(key):
    """
    Decorator to cache the result of the decorated function.  Uses the
    provided `key` function to compute the key for cache lookup. `key`
    receives all arguments passed to the function.
    """

    def deco(func):
        cache = {}

        @functools.wraps(func)
        def inner(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            if cache_key not in cache:
                result = func(*args, **kwargs)
                cache[cache_key] = result
            # `cache_key` *must* be in `cache`, use `.get()`
            # as it is faster:
            return cache.get(cache_key)

        return inner

    return deco

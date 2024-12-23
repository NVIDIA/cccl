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

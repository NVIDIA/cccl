# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import Any, Callable, Hashable

import numpy as np

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device


# Registry thet maps type -> key function for extracting cache key
# from a value of that type.
_KEY_FUNCTIONS: dict[type, Callable[[Any], Hashable]] = {}


def _key_for(value: Any) -> Hashable:
    """
    Extract a cache key from a value using the registered KEY_FUNCTIONS.

    This function checks the type of the value and delegates to the
    appropriate registered keyer. Falls back to using the value
    directly if no keyer is registered.

    Args:
        value: The value to extract a cache key from

    Returns:
        A hashable cache key
    """
    # Handle sequences (lists, tuples) by recursively converting to tuple
    if isinstance(value, (list, tuple)):
        return tuple(_key_for(item) for item in value)

    # Check for exact type match first
    value_type = type(value)
    if value_type in _KEY_FUNCTIONS:
        return _KEY_FUNCTIONS[value_type](value)

    # Check for instance match (handles inheritance)
    for registered_type, keyer in _KEY_FUNCTIONS.items():
        if isinstance(value, registered_type):
            return keyer(value)

    # For numpy dtypes, return directly (they're already hashable)
    if isinstance(value, np.dtype):
        return value

    # For None, return directly
    if value is None:
        return None

    # Check if it's a DeviceArrayLike (duck-typed via __cuda_array_interface__)
    if hasattr(value, "__cuda_array_interface__"):
        # It's a device array-like, extract dtype and fully-qualified type name
        from ._utils.protocols import get_dtype

        # Use fully-qualified type name (e.g., 'numpy.ndarray' v/s 'cupy.ndarray')
        type_fqn = f"{type(value).__module__}.{type(value).__name__}"
        return (type_fqn, get_dtype(value))

    # Check if it has a dtype attribute
    if hasattr(value, "dtype"):
        type_fqn = f"{type(value).__module__}.{type(value).__name__}"
        return (type_fqn, value.dtype)

    # For callables, wrap in CachableFunction
    if callable(value):
        return CachableFunction(value)

    # Fallback: use value directly (assumes it's hashable)
    return value


def _make_cache_key_from_args(*args, **kwargs) -> tuple:
    """
    Create a cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        A tuple containing the extracted cache keys
    """

    positional_keys = tuple(_key_for(arg) for arg in args)

    # Sort kwargs by key name for consistent ordering
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        kwarg_keys = tuple((k, _key_for(v)) for k, v in sorted_kwargs)
        return positional_keys + (kwarg_keys,)

    return positional_keys


# Register built-in types
# Include fully-qualified type name to distinguish np.ndarray from cp.ndarray from GpuStruct
_KEY_FUNCTIONS[np.ndarray] = lambda arr: ("numpy.ndarray", arr.dtype)


# ============================================================================
# Central Cache Registry
# ============================================================================

# Central registry of all algorithm caches
_cache_registry: dict[str, object] = {}


class _CacheWithKeyFunctionsDecorator:
    """
    Decorator class that provides caching with automatic key extraction.
    """

    @staticmethod
    def register(type_, key_function):
        """
        Register a key function for a specific type.

        A key function is a function that extracts a hashable cache key from a value.

        Args:
            type_: The type to register
            key_function: A callable that takes an instance of type_ and returns a hashable cache key
        """
        _KEY_FUNCTIONS[type_] = key_function

    def __call__(self, func_or_key_function=None):
        """
        Decorator to cache the result of the decorated function.

        If `key` is provided, it should be a function that computes the cache key
        from the function arguments. Otherwise, the cache key is automatically
        computed from all arguments using the cache key registry.

        Notes
        -----
        The CUDA compute capability of the current device is appended to
        the cache key.

        The decorated function is automatically registered in the central
        cache registry for easy cache management.

        Args:
            func_or_key_function: The function to decorate, or a custom key function.
        """
        # Check if this is being used without parentheses (@cache_with_registered_key_functions)
        if callable(func_or_key_function) and not hasattr(
            func_or_key_function, "__self__"
        ):
            # Direct decoration: @cache_with_registered_key_functions
            # func_or_key is the actual function being decorated
            return self._make_wrapper(None)(func_or_key_function)

        # Otherwise, return a decorator (supports both @cache_with_registered_key_functions() and @cache_with_registered_key_functions(key_fn))
        key = func_or_key_function  # Could be None or a custom key function
        return self._make_wrapper(key)

    def _make_wrapper(self, key_function):
        """
        Create the actual caching wrapper.

        Args:
            key_function: Optional function to compute cache key from args/kwargs.
                 If None, uses the cache key registry to automatically compute keys.
        """

        def deco(func):
            cache = {}

            @functools.wraps(func)
            def inner(*args, **kwargs):
                cc = Device().compute_capability
                if key_function is not None:
                    # Use provided key function
                    user_cache_key = key_function(*args, **kwargs)
                else:
                    # Use registry-based automatic key generation
                    user_cache_key = _make_cache_key_from_args(*args, **kwargs)

                cache_key = (user_cache_key, tuple(cc))
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


# Create the singleton instance
cache_with_registered_key_functions = _CacheWithKeyFunctionsDecorator()

# Keep old name for backwards compatibility
cache_with_key = cache_with_registered_key_functions


def _hash_device_array_like(value):
    # hash based on pointer, shape, and dtype
    ptr = value.__cuda_array_interface__["data"][0]
    shape = value.__cuda_array_interface__["shape"]
    dtype = value.__cuda_array_interface__["typestr"]
    return hash((ptr, shape, dtype))


def _make_hashable(value):
    import numba.cuda.dispatcher

    from .typing import DeviceArrayLike

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

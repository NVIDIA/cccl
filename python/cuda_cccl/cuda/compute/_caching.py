# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import types
from typing import Any, Callable, Hashable

import numpy as np

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device

from ._utils.protocols import get_dtype, get_shape, is_device_array
from .typing import DeviceArrayLike, GpuStruct

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

    # DeviceArrayLike is not a runtime-checkable protocol, so
    # we cannot isinstance() with it.
    if is_device_array(value):
        return _KEY_FUNCTIONS[DeviceArrayLike](value)

    # Check for instance match (handles inheritance)
    for registered_type, keyer in _KEY_FUNCTIONS.items():
        if registered_type is not DeviceArrayLike and isinstance(
            value, registered_type
        ):
            return keyer(value)

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


# Central registry of all algorithm caches
_cache_registry: dict[str, object] = {}


class _CacheWithRegisteredKeyFunctions:
    """
    Decorator to cache the result of the decorated function.

    The cache key is automatically computed from the decorated function's
    arguments using the registered key functions.
    """

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to cache the result of the decorated function.

        Args:
            func: The function whose result is to be cached.

        Notes
        -----
        The CUDA compute capability of the current device is appended to
        the cache key.
        """
        cache: dict = {}

        @functools.wraps(func)
        def inner(*args, **kwargs):
            cc = Device().compute_capability
            user_cache_key = _make_cache_key_from_args(*args, **kwargs)
            cache_key = (user_cache_key, tuple(cc))
            if cache_key not in cache:
                result = func(*args, **kwargs)
                cache[cache_key] = result
            return cache[cache_key]

        inner.cache_clear = cache.clear  # type: ignore[attr-defined]

        # Register the cache in the central registry
        _cache_registry[func.__qualname__] = inner

        return inner

    def register(self, type_: type, key_function: Callable[[Any], Hashable]) -> None:
        """
        Register a key function for a specific type.

        A key function extracts a hashable cache key from a value.

        Args:
            type_: The type to register
            key_function: A callable that takes an instance of type_ and
                returns a hashable cache key
        """
        _KEY_FUNCTIONS[type_] = key_function


cache_with_registered_key_functions = _CacheWithRegisteredKeyFunctions()


def _make_hashable(value):
    # duck-type check for numba.cuda.CUDADispatcher:
    if hasattr(value, "py_func") and callable(value.py_func):
        return CachableFunction(value.py_func)
    elif is_device_array(value):
        # Ops with device arrays in globals/closures will be handled
        # by stateful op machinery, which enables updating the state
        # (pointers). Thus, we only cache on the dtype and shape of
        # the referenced array, but not its pointer.
        return (get_dtype(value), get_shape(value))
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

    # TODO: eventually, move this class to _jit.py as it only
    # has to do with caching of Python callables that will be
    # JIT compiled.
    def __init__(self, func):
        self._func = func

        closure = func.__closure__ if func.__closure__ is not None else []
        contents = []
        # Make closure contents hashable
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


# Register keyers for built-in types
# Include fully-qualified type name to distinguish np.ndarray from cp.ndarray from GpuStruct
def _type_fqn(v):
    return f"{type(v).__module__}.{type(v).__name__}"


cache_with_registered_key_functions.register(
    np.ndarray, lambda arr: ("numpy.ndarray", arr.dtype)
)
cache_with_registered_key_functions.register(
    types.FunctionType, lambda fn: CachableFunction(fn)
)
cache_with_registered_key_functions.register(
    DeviceArrayLike, lambda v: (_type_fqn(v), get_dtype(v))
)
cache_with_registered_key_functions.register(
    GpuStruct, lambda v: (_type_fqn(v), v.dtype)
)

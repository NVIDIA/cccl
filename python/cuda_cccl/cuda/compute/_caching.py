# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import functools
import os
import types
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device

from ._utils.protocols import get_dtype, get_shape, is_device_array
from .struct import _Struct

# Registry thet maps type -> key function for extracting cache key
# from a value of that type.
_KEY_FUNCTIONS: dict[type, Callable[[Any], Hashable]] = {}


def _type_fqn(v):
    # fully-qualified type name to distinguish np.ndarray from cp.ndarray from GpuStruct
    return f"{type(v).__module__}.{type(v).__name__}"


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
        return (_type_fqn(value), get_dtype(value))

    # Check for instance match (handles inheritance)
    for registered_type, keyer in _KEY_FUNCTIONS.items():
        if isinstance(value, registered_type):
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

# Optional on-disk cache directory (None = disabled)
_disk_cache_dir: Path | None = None
# Pick up env var at import time
_env_dir = os.environ.get("CUDA_COMPUTE_CACHE_DIR")
if _env_dir:
    _disk_cache_dir = Path(_env_dir)


def set_cache_dir(path) -> None:
    """Enable or disable the transparent on-disk build cache.

    When set, compiled algorithm objects are automatically persisted to *path*
    and reloaded from disk on subsequent calls with the same arguments, avoiding
    NVRTC/nvJitLink recompilation across Python processes.

    Args:
        path: Directory to store ``.alg`` cache files, or ``None`` to disable.
    """
    global _disk_cache_dir
    _disk_cache_dir = Path(path) if path is not None else None
    if _disk_cache_dir is not None:
        _disk_cache_dir.mkdir(parents=True, exist_ok=True)


def _result_algorithm_name(result) -> str | None:
    """Return the algorithm tag string for a build result, or None."""
    from .algorithms._reduce import _Reduce
    from .algorithms._sort._merge_sort import _MergeSort

    if isinstance(result, _Reduce):
        return "reduce"
    if isinstance(result, _MergeSort):
        return "merge_sort"
    return None


def _disk_cache_path(cache_key: tuple) -> Path | None:
    """Return the disk-cache file path for *cache_key*, or None if cache is disabled."""
    if _disk_cache_dir is None:
        return None
    import hashlib

    key_hash = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:24]
    return _disk_cache_dir / f"{key_hash}.alg"


def _load_from_disk(path: Path):
    """Load an algorithm object from *path*. Returns None on any error."""
    try:
        from ._serialization import load_algorithm

        return load_algorithm(path)
    except Exception:
        return None


def _save_to_disk(path: Path, result) -> None:
    """Save *result* to *path* if it supports ``.save()``."""
    try:
        if hasattr(result, "save"):
            result.save(path)
    except Exception:
        pass


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
            cache_key = (func.__qualname__, user_cache_key, tuple(cc))
            if cache_key not in cache:
                disk_path = _disk_cache_path(cache_key)
                if disk_path is not None and disk_path.exists():
                    loaded = _load_from_disk(disk_path)
                    if loaded is not None:
                        cache[cache_key] = loaded
                        return loaded
                result = func(*args, **kwargs)
                cache[cache_key] = result
                if disk_path is not None:
                    _save_to_disk(disk_path, result)
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
    recompilation on the next invocation. Also removes any ``.alg`` files
    from the on-disk cache directory (if configured).

    Example
    -------
    >>> import cuda.compute
    >>> cuda.compute.clear_all_caches()
    """
    for cached_func in _cache_registry.values():
        cached_func.cache_clear()
    if _disk_cache_dir is not None and _disk_cache_dir.is_dir():
        for f in _disk_cache_dir.glob("*.alg"):
            try:
                f.unlink()
            except OSError:
                pass


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
                # if `name` is found in __globals__, try and hash
                # the referenced object. If `name` is not found in
                # __globals__, (e.g., `name` is part of a dotted
                # name like `np.argmax`), for caching purposes we
                # use the hash of the name itself. Assumes numba
                # known how to interpret the dotted name at JIT
                # time.
                _make_hashable(func.__globals__.get(name, name))
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

cache_with_registered_key_functions.register(
    np.ndarray, lambda arr: ("numpy.ndarray", arr.dtype)
)
cache_with_registered_key_functions.register(
    types.FunctionType, lambda fn: CachableFunction(fn)
)
cache_with_registered_key_functions.register(_Struct, lambda v: (_type_fqn(v), v.dtype))
# Numpy scalars: key by dtype only so np.float32(0) and np.float32(1) hit the same entry.
cache_with_registered_key_functions.register(
    np.generic, lambda v: ("numpy.scalar", np.dtype(type(v)))
)


def _register_proxy_types():
    from ._proxy import ProxyArray, ProxyValue

    cache_with_registered_key_functions.register(
        ProxyArray, lambda v: ("ProxyArray", v.dtype)
    )
    cache_with_registered_key_functions.register(
        ProxyValue, lambda v: ("ProxyValue", v.dtype)
    )


_register_proxy_types()

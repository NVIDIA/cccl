# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import functools
import threading
import types
import weakref
from typing import Any, Callable, Hashable

import numpy as np

from cuda.core import Device

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


class _ThreadLocalCaches:
    """
    Container for wrapper caches owned by a single Python thread.

    Each thread gets its own instance via ``threading.local()``. We use
    ``__weakref__`` to enable the process-wide registry of caches to hold weak
    references to the thread's caches. That way, if a thread exits, its caches
    will be garbage collected and removed from the registry even if the
    process-wide registry still references them.
    """

    __slots__ = ("wrapper_caches", "__weakref__")

    def __init__(self) -> None:
        # Outer key: decorated algorithm factory name. Inner key: current thread
        # id, current CUDA runtime device ordinal, compute capability, and
        # specialization key derived from factory arguments.
        self.wrapper_caches: dict[str, dict[Hashable, Any]] = {}


class _InFlightBuild:
    """
    Coordination state for one shared build-result currently being built.

    The first thread for a cache key runs the builder. Other threads wait on
    ``condition`` and receive either the completed build result or the builder's
    exception.
    """

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.done = False
        self.result: Any = None
        self.exception: BaseException | None = None


_thread_local = threading.local()
# Process wide registry of per-thread caches. It enables a thread to call
# clear_all_caches() to clear all caches across all threads.
_thread_cache_registry: weakref.WeakSet[_ThreadLocalCaches] = weakref.WeakSet()
_thread_cache_registry_lock = threading.Lock()

_shared_build_cache: dict[Hashable, Any] = {}
_in_flight_builds: dict[Hashable, _InFlightBuild] = {}
_shared_build_cache_lock = threading.Lock()


def _get_current_device_info() -> tuple[int, tuple[int, int]]:
    device = Device()
    cc = device.compute_capability
    return device.device_id, (cc.major, cc.minor)


def _get_thread_caches() -> _ThreadLocalCaches:
    caches = getattr(_thread_local, "caches", None)
    if caches is None:
        caches = _ThreadLocalCaches()
        _thread_local.caches = caches
        with _thread_cache_registry_lock:
            _thread_cache_registry.add(caches)
    return caches


def _clear_wrapper_caches(cache_name: str | None = None) -> None:
    with _thread_cache_registry_lock:
        thread_caches = list(_thread_cache_registry)

    for caches in thread_caches:
        if cache_name is None:
            caches.wrapper_caches.clear()
        else:
            caches.wrapper_caches.pop(cache_name, None)


def cache_build_result(
    build_result_type: type,
    *key_args,
    builder: Callable[[], Any],
) -> Any:
    """
    Cache a shared Cython build-result object for the current CUDA device.

    The key intentionally excludes the current Python thread. Wrappers are
    cached per thread, but build results are shared across threads for the same
    device ordinal and specialization key.

    Args:
        build_result_type: Cython build-result type. This separates different
            build-result caches that may otherwise have identical specialization
            keys.
        *key_args: Positional values used to form the specialization part of
            the cache key.
        builder: Callable that creates the build result on a cache miss.
            Exactly one thread runs this callable for a given key while other
            threads wait for the result.

    Returns:
        The cached or newly built Cython build-result object.
    """
    device_id, cc_key = _get_current_device_info()
    user_cache_key = _make_cache_key_from_args(*key_args)
    cache_key = (build_result_type, device_id, cc_key, user_cache_key)

    with _shared_build_cache_lock:
        if cache_key in _shared_build_cache:
            return _shared_build_cache[cache_key]

        in_flight = _in_flight_builds.get(cache_key)
        if in_flight is None:
            in_flight = _InFlightBuild()
            _in_flight_builds[cache_key] = in_flight
            is_builder = True
        else:
            is_builder = False

    if is_builder:
        try:
            result = builder()
        except BaseException as exc:
            with _shared_build_cache_lock:
                _in_flight_builds.pop(cache_key, None)
            with in_flight.condition:
                in_flight.exception = exc
                in_flight.done = True
                in_flight.condition.notify_all()
            raise

        with _shared_build_cache_lock:
            _shared_build_cache[cache_key] = result
            _in_flight_builds.pop(cache_key, None)
        with in_flight.condition:
            in_flight.result = result
            in_flight.done = True
            in_flight.condition.notify_all()
        return result

    with in_flight.condition:
        while not in_flight.done:
            in_flight.condition.wait()
        if in_flight.exception is not None:
            raise in_flight.exception
        return in_flight.result


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
        cache_name = func.__qualname__

        @functools.wraps(func)
        def inner(*args, **kwargs):
            device_id, cc_key = _get_current_device_info()
            user_cache_key = _make_cache_key_from_args(*args, **kwargs)
            cache_key = (threading.get_ident(), device_id, cc_key, user_cache_key)
            thread_caches = _get_thread_caches()
            cache = thread_caches.wrapper_caches.setdefault(cache_name, {})
            if cache_key not in cache:
                result = func(*args, **kwargs)
                cache[cache_key] = result
            return cache[cache_key]

        inner.cache_clear = lambda: _clear_wrapper_caches(cache_name)  # type: ignore[attr-defined]

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
    _clear_wrapper_caches()
    with _shared_build_cache_lock:
        _shared_build_cache.clear()
        _in_flight_builds.clear()


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

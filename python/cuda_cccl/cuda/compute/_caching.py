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

try:
    from cuda.core._utils.cuda_utils import CUDAError
except ImportError:
    from cuda.core.experimental._utils.cuda_utils import CUDAError

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


# Process-wide registry of all algorithm caches.
_process_wide_cache_registry: dict[str, object] = {}


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
        # Outer key: decorated algorithm factory name. Inner key: target
        # identity and specialization key derived from factory arguments. The
        # target is a device ordinal plus compute capability for default
        # builds, or the explicit AOT target otherwise. Thread isolation comes
        # from the threading.local container itself, not from the key.
        self.wrapper_caches: dict[str, dict[Hashable, Any]] = {}


class _InFlightBuild:
    """
    Coordination state for one shared build-result currently being built.

    The first thread for a cache key runs the builder. Other threads wait on
    ``event`` and receive either the completed build result or the builder's
    exception.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.exception: BaseException | None = None


_thread_local = threading.local()
# Process wide registry of per-thread caches. It enables a thread to call
# clear_all_caches() to clear all caches across all threads.
_process_wide_thread_cache_registry: weakref.WeakSet[_ThreadLocalCaches] = (
    weakref.WeakSet()
)
_process_wide_thread_cache_registry_lock = threading.Lock()

# Values are either completed build-result collections or temporary
# _InFlightBuild entries.
_process_wide_build_results_cache: dict[Hashable, Any] = {}
# First successful default-build collection per (build-result type, compute
# capability, specialization). The compiled payload only depends on the
# compute capability, so a default build on another same-cc device ordinal
# clones the donor's payload instead of running a full native compilation.
_process_wide_default_build_donors: dict[Hashable, Any] = {}
_CACHE_MISS = object()


def _cache_single_flight(
    cache: dict[Hashable, Any], cache_key: Hashable, builder: Callable[[], Any]
) -> Any:
    """Return a cached value, coalescing concurrent builds for the same key."""
    cache_entry = cache.get(cache_key, _CACHE_MISS)
    if cache_entry is _CACHE_MISS:
        in_flight = _InFlightBuild()
        # setdefault elects one builder without an explicit lock on cache hits.
        cache_entry = cache.setdefault(cache_key, in_flight)
        if cache_entry is in_flight:
            try:
                result = builder()
                in_flight.result = result
                cache[cache_key] = result
            except BaseException as exc:
                in_flight.exception = exc
                cache.pop(cache_key, None)
                raise
            finally:
                in_flight.event.set()
            return result

    if isinstance(cache_entry, _InFlightBuild):
        cache_entry.event.wait()
        if cache_entry.exception is not None:
            raise cache_entry.exception
        return cache_entry.result

    return cache_entry


class _BuildResultCollection(dict[int, Any]):
    """Compiled build results with process-shared, per-device loaded state.

    The mapping contains one canonical native build result per compute
    capability. Explicit AOT results begin unloaded and can be shared across
    threads without a current device. On first use, one device claims the
    canonical result; additional devices clone it through serialization before
    loading, so compiled payloads are reused without sharing device-owned native
    handles.
    """

    def __init__(
        self,
        build_results: dict[int, Any],
        *,
        loaded_device_id: int | None = None,
    ) -> None:
        super().__init__(build_results)
        self._owner_devices: dict[int, int] = {}
        self._loaded_results: dict[Hashable, Any] = {}
        self._device_bound_result: Any | None = None
        # Loading the canonical result mutates its native handle fields, while
        # cloning serializes its payload. Serialize those source operations, but
        # keep completed-result lookups lock-free.
        self._source_locks = {cc: threading.Lock() for cc in self}

        if loaded_device_id is not None:
            if len(self) != 1:
                raise ValueError(
                    "A device-bound build-result collection must be singular"
                )
            for cc, build_result in self.items():
                self._owner_devices[cc] = loaded_device_id
                self._loaded_results[(cc, loaded_device_id)] = build_result
                self._device_bound_result = build_result

    def resolve(self, cc: int, device_id: int) -> Any:
        """Return the build result loaded for ``device_id`` without recompiling."""
        # Completed loads are terminal — never removed or replaced — so the
        # warm path is one lock-free lookup with no dict mutation and no
        # closure allocation. Misses and in-flight loads (which only exist
        # around the first load per device) fall through to the single-flight
        # machinery below.
        loaded = self._loaded_results.get((cc, device_id))
        if loaded is not None and not isinstance(loaded, _InFlightBuild):
            return loaded

        source = self[cc]
        owner_device = self._owner_devices.setdefault(cc, device_id)

        def load_for_device():
            if owner_device == device_id:
                with self._source_locks[cc]:
                    source.load()
                return source

            with self._source_locks[cc]:
                blob = source.serialize()
            result = type(source).deserialize(blob, load=False, check_cc=True)
            result.load()
            return result

        return _cache_single_flight(
            self._loaded_results, (cc, device_id), load_for_device
        )

    def serialize_build_result(self, cc: int) -> bytes:
        """Serialize a canonical result without racing its first device load."""
        with self._source_locks[cc]:
            return self[cc].serialize()


def _get_current_device_info() -> tuple[int, tuple[int, int]]:
    device = Device()
    cc_major, cc_minor = device.compute_capability
    return device.device_id, (cc_major, cc_minor)


def _get_thread_caches() -> _ThreadLocalCaches:
    caches = getattr(_thread_local, "caches", None)
    if caches is None:
        caches = _ThreadLocalCaches()
        _thread_local.caches = caches
        with _process_wide_thread_cache_registry_lock:
            _process_wide_thread_cache_registry.add(caches)
    return caches


def _clear_wrapper_caches(cache_name: str | None = None) -> None:
    with _process_wide_thread_cache_registry_lock:
        thread_caches = list(_process_wide_thread_cache_registry)

    for caches in thread_caches:
        if cache_name is None:
            caches.wrapper_caches.clear()
        else:
            caches.wrapper_caches.pop(cache_name, None)


def cache_build_results(
    build_result_type: type,
    *key_args,
    compute_capability,
    builder: Callable[[], Any],
) -> Any:
    """
    Cache a shared collection of Cython build-result objects.

    Current-device builds are keyed by CUDA device ordinal and compute
    capability; the compiled payload, however, is shared across same-cc
    ordinals — the first successful build per specialization and compute
    capability becomes a payload donor that later same-cc devices clone
    through serialization instead of recompiling. Explicit AOT builds have no
    current device and are instead keyed by their normalized target compute
    capabilities. The key intentionally
    excludes the current Python thread so wrappers can share compiled results.

    Args:
        build_result_type: Cython build-result type. This separates collections
            that may otherwise have identical specialization keys.
        *key_args: Positional values used to form the specialization part of
            the cache key.
        compute_capability: Explicit AOT target or ``None`` for the current
            device.
        builder: Callable that creates the build-result collection on a cache miss.
            Exactly one thread runs this callable for a given key while other
            threads wait for the result.

    Returns:
        The cached or newly built build-result collection.
    """
    if compute_capability is None:
        # The factory decorator already queried the device on the wrapper-cache
        # miss path and hands the result through thread-local state; fall back
        # to a fresh query for direct construction (e.g. deserialization).
        device_info = getattr(_thread_local, "factory_device_info", None)
        if device_info is None:
            device_info = _get_current_device_info()
        device_id, cc_key = device_info
        target_key: Hashable = ("device", device_id, cc_key)
        user_cache_key = _make_cache_key_from_args(*key_args)
        cache_key = (build_result_type, target_key, user_cache_key)
        donor_key = (build_result_type, cc_key, user_cache_key)

        def build_or_clone():
            donor = _process_wide_default_build_donors.get(donor_key)
            if donor is not None:
                cloned = _clone_collection_for_device(donor, device_id)
                if cloned is not None:
                    return cloned
            collection = builder()
            _process_wide_default_build_donors.setdefault(donor_key, collection)
            return collection

        return _cache_single_flight(
            _process_wide_build_results_cache, cache_key, build_or_clone
        )

    from ._cccl_interop import cc_to_key, normalize_compute_capabilities

    target_ccs = normalize_compute_capabilities(compute_capability)
    assert target_ccs is not None
    target_key = ("aot", tuple(cc_to_key(cc) for cc in target_ccs))
    user_cache_key = _make_cache_key_from_args(*key_args)
    cache_key = (build_result_type, target_key, user_cache_key)
    return _cache_single_flight(_process_wide_build_results_cache, cache_key, builder)


def _clone_collection_for_device(donor, device_id: int):
    """Clone a same-cc donor's compiled payload for ``device_id``.

    Default builds are keyed per device ordinal, but their compiled payload
    only depends on the compute capability. Reconstructing the donor through
    serialize -> deserialize(load=False) -> load costs milliseconds where a
    full native build costs around a second. Returns ``None`` when cloning is
    unavailable — most notably on backends without build-result serialization
    (the v2 HostJIT backend today) — so the caller falls back to a full
    build. ``check_cc`` re-validates the payload against the current device.
    """
    try:
        (packed_cc,) = donor
        blob = donor.serialize_build_result(packed_cc)
        source = donor[packed_cc]
        clone = type(source).deserialize(blob, load=False, check_cc=True)
        clone.load()
    except Exception:
        return None
    return _BuildResultCollection({packed_cc: clone}, loaded_device_id=device_id)


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
        Default builds append the current CUDA device and compute capability to
        the cache key. Explicit AOT builds include their normalized target
        compute capabilities without querying a device.
        """
        cache_name = func.__qualname__

        @functools.wraps(func)
        def inner(*args, **kwargs):
            user_cache_key = _make_cache_key_from_args(*args, **kwargs)
            # When the caller targets explicit compute capabilities, that value
            # is already part of user_cache_key (it arrives as a kwarg) and we
            # must NOT query a device — the whole point is to build without a
            # GPU. Otherwise, salt the key with the current device's cc so a
            # build cached on one device isn't reused on another.
            if kwargs.get("compute_capability") is None:
                # Only device-availability failures should be reinterpreted as
                # "pass compute_capability": no driver / no device raises
                # CUDAError, and querying device 0 on a machine with zero
                # devices raises ValueError. Anything else (a real bug) must
                # propagate untouched. The original error is chained and echoed
                # so a genuine driver/permission failure isn't hidden behind a
                # misleading "no device" message.
                try:
                    device_id, cc = _get_current_device_info()
                except (CUDAError, ValueError) as e:
                    raise RuntimeError(
                        "make_<algo> was called without compute_capability and the "
                        f"current CUDA device could not be queried ({e}). Pass "
                        "compute_capability=<cc or list of ccs> to compile without "
                        "a GPU (e.g. with ProxyArray / ProxyValue)."
                    ) from e
                target_cc_arg = cc
            else:
                device_id = None
                cc = None
                target_cc_arg = kwargs.get("compute_capability")
            # No thread id in the key: the containing cache is threading.local,
            # so each thread only ever sees its own entries.
            cache_key = (device_id, cc, user_cache_key)
            thread_caches = _get_thread_caches()
            cache = thread_caches.wrapper_caches.setdefault(cache_name, {})
            if cache_key not in cache:
                # Shared device code (operators, iterators) is compiled to LTO-IR
                # once and linked into every per-arch build result, so it must target
                # the lowest requested cc (nvJitLink requires final SM >= each
                # linked input's arch). Set that target around the build.
                from ._target_cc import target_cc

                # Hand the device info queried above to cache_build_results
                # (reached through the wrapper's __init__) so the miss path
                # does not construct a second cuda.core Device. Saved/restored
                # so nested factory calls fall back to their own query.
                previous_device_info = getattr(
                    _thread_local, "factory_device_info", None
                )
                _thread_local.factory_device_info = (
                    (device_id, cc) if device_id is not None else None
                )
                try:
                    with target_cc(target_cc_arg):
                        result = func(*args, **kwargs)
                finally:
                    _thread_local.factory_device_info = previous_device_info
                cache[cache_key] = result
            return cache[cache_key]

        inner.cache_clear = lambda: _clear_wrapper_caches(cache_name)  # type: ignore[attr-defined]

        # Register the cache in the central registry
        _process_wide_cache_registry[func.__qualname__] = inner

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
    elif isinstance(value, (np.number, np.bool_)):
        return ("numpy.scalar", value.dtype.str, value.tobytes())
    elif isinstance(value, (bool, int, float)):
        # Python scalars are immutable values; key them by type and value so
        # equal-valued scalars share a cache entry. Without this they fall
        # through to ``id(value)`` below, and a fresh (non-interned) ``int``/
        # ``float`` with the same value misses the build cache on every call.
        # ``_type_fqn`` keeps ``True`` distinct from ``1``/``1.0`` (and avoids
        # collisions between like-named scalar subclasses from other modules).
        return ("python.scalar", _type_fqn(value), value)
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

    This function clears cached algorithm wrappers and completed build results
    in the current process, forcing recompilation on the next invocation.
    Useful for benchmarking compilation time.

    This function is not synchronized with active factory calls or algorithm
    execution. Callers that use it in a multi-threaded program must externally
    synchronize with all threads that may create or use cuda.compute algorithm
    objects. If a build is already in progress, that build may complete after
    this function returns and repopulate the completed build-result cache.

    Example
    -------
    >>> import cuda.compute
    >>> cuda.compute.clear_all_caches()
    """
    _clear_wrapper_caches()
    _process_wide_build_results_cache.clear()
    _process_wide_default_build_donors.clear()
    # Auxiliary caches registered process-wide (e.g. _jit._infer_return_type)
    # must be cleared too, so builds after a clear really are cold. Factory
    # entries' cache_clear is idempotent with _clear_wrapper_caches above.
    for cached_func in _process_wide_cache_registry.values():
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


def _register_proxy_types():
    # Registered lazily to avoid importing _proxy (and numpy-dtype construction)
    # at module import time; the keys are dtype-only so equal-dtype proxies share
    # a cache entry.
    from ._proxy import ProxyArray, ProxyValue

    cache_with_registered_key_functions.register(
        ProxyArray, lambda v: ("ProxyArray", v.dtype)
    )
    cache_with_registered_key_functions.register(
        ProxyValue, lambda v: ("ProxyValue", v.dtype)
    )


_register_proxy_types()

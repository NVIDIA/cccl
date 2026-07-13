# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import functools
import threading
import types
import weakref
from typing import Any, Callable, Hashable, NamedTuple, TypeVar

import numpy as np

from cuda.core import Device

try:
    from cuda.core._utils.cuda_utils import CUDAError
except ImportError:
    from cuda.core.experimental._utils.cuda_utils import CUDAError

from ._utils.protocols import get_dtype, get_shape, is_device_array
from .struct import _Struct

try:
    from ._build_info import USING_V2  # type: ignore[import-not-found]
except ImportError:
    USING_V2 = False

# Whether the backend can serialize build results (the v2 HostJIT backend
# cannot, today). Serialization is what lets same-cc devices share one default
# build — a non-owner device loads by cloning the shared entry through
# serialize -> deserialize -> load — so backends without it key default builds
# per device ordinal and build independently per device instead.
# TODO: delete this flag (and every branch on it) once v2 supports
# build-result serialization.
_BACKEND_SERIALIZES_BUILD_RESULTS = not USING_V2

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


# The specialization part of every cache key: what _make_cache_key_from_args
# extracts from the user-facing arguments (dtypes, op identity, iterator
# kinds, ...).
_SpecializationKey = tuple[Hashable, ...]


def _make_cache_key_from_args(*args, **kwargs) -> _SpecializationKey:
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
        # Outer key: decorated algorithm factory name, e.g.,
        # make_reduce_into.__qualname__. Inner key: _WrapperCacheKey. No
        # thread id is needed in either key because each thread holds a
        # separate thread local object of this class.
        self.wrapper_caches: dict[str, dict[_WrapperCacheKey, Any]] = {}


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


# A compute capability packed into one int as major * 10 + minor, e.g.
# (9, 0) -> 90 and (12, 0) -> 120; cc_to_key / key_to_cc in _cccl_interop
# convert to and from the (major, minor) pair form. Purely documentary:
# mypy treats it as int.
_PackedCCKey = int


class _DeviceBuildTarget(NamedTuple):
    """
    Target identity of a per-device wrapper or build entry.

    Two roles. In the wrapper cache it keys every default-build wrapper:
    wrappers hold device-bound state (their construction-time binding), so
    each device gets its own. In the build-results cache it is used only when
    the backend cannot serialize build results (the v2 HostJIT backend
    today): sharing one entry across same-cc devices requires cloning it
    through serialization, so such backends key default builds per device
    ordinal instead.
    TODO: once v2 supports build-result serialization, delete the build-cache
    role (the _BACKEND_SERIALIZES_BUILD_RESULTS branch in
    cache_build_results); the wrapper-cache role remains.

    NamedTuples compare as plain tuples, so all target kinds must keep
    structurally disjoint layouts (arity or element types) to never compare
    equal to each other.
    """

    device_id: int
    cc: tuple[int, int]


class _DefaultBuildTarget(NamedTuple):
    """
    Target identity of a default build shared across same-cc devices.

    Holds the packed cc alone: the compiled payload depends only on the cc,
    and _PerCCBuildResults.resolve() gives each device its own loaded state.
    See _DeviceBuildTarget for the cross-kind equality constraint.
    """

    cc_key: _PackedCCKey


class _AOTBuildTarget(NamedTuple):
    """
    Target identity of an explicit AOT build, with no device attached.

    Holds the normalized, sorted, packed compute-capability keys. See
    _DeviceBuildTarget for the cross-kind equality constraint.
    """

    cc_keys: tuple[_PackedCCKey, ...]


# Inner key of a thread's per-factory wrapper cache; see _ThreadLocalCaches.
# The target is None for explicit AOT builds: the decorator performs no device
# query or cc normalization there, and the raw compute_capability kwarg is
# already part of the specialization. Differently spelled ccs (80 vs (8, 0))
# therefore yield distinct wrappers, which still share one compiled build
# because cache_build_results normalizes its own key.
_WrapperCacheKey = tuple[_DeviceBuildTarget | None, _SpecializationKey]
# Composite key of the process-wide build-results cache:
# (build-result type, build target (device or AOT), specialization). The
# build-result type is the algorithm's Cython class from _bindings (e.g.
# DeviceReduceBuildResult), namespacing entries per algorithm.
# TODO: drop _DeviceBuildTarget from this union once v2 supports build-result
# serialization; it then only keys the wrapper cache.
_BuildResultsCacheKey = tuple[
    type,
    _DefaultBuildTarget | _DeviceBuildTarget | _AOTBuildTarget,
    _SpecializationKey,
]


_thread_local = threading.local()
# Process wide registry of per-thread caches. It enables a thread to call
# clear_all_caches() to clear all caches across all threads.
_process_wide_thread_cache_registry: weakref.WeakSet[_ThreadLocalCaches] = (
    weakref.WeakSet()
)
_process_wide_thread_cache_registry_lock = threading.Lock()

# _InFlightBuild entries are temporary: replaced by the completed build
# results or removed on builder failure.
_process_wide_build_results_cache: dict[
    _BuildResultsCacheKey, _PerCCBuildResults | _InFlightBuild
] = {}
_CACHE_MISS = object()

_KeyT = TypeVar("_KeyT", bound=Hashable)


def _cache_single_flight(
    cache: dict[_KeyT, Any], cache_key: _KeyT, builder: Callable[[], Any]
) -> Any:
    """Return a cached value, coalescing concurrent builds for the same key.

    ``cache`` may be any single-flight dict — currently the process-wide
    build-results cache and the per-device loaded results inside each
    _PerCCBuildResults. Its entries are one of two things: a completed value,
    which is terminal, or a temporary _InFlightBuild while the one elected
    caller runs ``builder``; other callers wait on its event and receive the
    same result or exception. A failed builder's entry is removed so a later
    call retries.
    """
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


class _PerCCBuildResults(dict[_PackedCCKey, Any]):
    """One algorithm specialization's compiled build results, keyed by target cc.

    Instances may be shared process-wide across threads: the factory build
    cache hands all same-specialization wrappers one instance. The compiled
    payload depends only on the cc, but a loaded build result holds
    device-specific native state, so devices never share one. The first
    device to execute claims and loads the canonical build result in place;
    each additional device lazily loads its own clone of the compiled payload
    (serialize -> deserialize -> load). This class tracks the loaded result —
    canonical or clone — assigned to each device.
    """

    def __init__(
        self,
        build_results: dict[_PackedCCKey, Any],
        *,
        loaded_device_id: int | None = None,
    ) -> None:
        super().__init__(build_results)
        # The single device that claimed each cc's canonical result and loads
        # it in place; every other device clones instead. The atomic claim
        # (resolve()'s setdefault) is what prevents double-loading the
        # canonical object.
        self._owner_devices: dict[_PackedCCKey, int] = {}
        # The loaded result each (cc key, device ordinal) pair executes: the
        # canonical result for its owner device; an independent clone — or,
        # when cloning fails, an independently built result — for every other
        # device. Also holds temporary _InFlightBuild entries while a first
        # load is in flight.
        self._loaded_results: dict[tuple[_PackedCCKey, int], Any] = {}
        # Loading the canonical result mutates its native handle fields, while
        # cloning serializes its payload. Serialize those source operations, but
        # keep completed-result lookups lock-free.
        self._source_locks = {cc: threading.Lock() for cc in self}

        if loaded_device_id is not None:
            # The caller already built and loaded the single entry on
            # loaded_device_id: pre-record the post-conditions resolve()'s
            # owner-load path would otherwise produce on first use.
            if len(self) != 1:
                raise ValueError("A device-bound _PerCCBuildResults must be singular")
            for cc, build_result in self.items():
                self._owner_devices[cc] = loaded_device_id
                self._loaded_results[(cc, loaded_device_id)] = build_result

    def resolve(self, cc: _PackedCCKey, device_id: int) -> Any:
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

    def serialize_build_result(self, cc: _PackedCCKey) -> bytes:
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
    Cache the shared Cython build results for one specialization.

    Current-device builds are keyed by compute capability alone and shared
    across same-cc device ordinals: the compiled payload only depends on the
    cc, and _PerCCBuildResults.resolve() gives each device its own loaded
    state. When the backend cannot serialize build results (the v2 HostJIT
    backend today), a non-owner device has no way to load the shared entry —
    loading it clones through serialization — so default builds are keyed per
    device ordinal instead and each device builds its own entry. Explicit AOT
    builds have no current device and are keyed by their normalized target
    compute capabilities. The key intentionally excludes the current Python
    thread so wrappers can share compiled results.

    Args:
        build_result_type: Cython build-result type. This separates entries
            that may otherwise have identical specialization keys.
        *key_args: Positional values used to form the specialization part of
            the cache key.
        compute_capability: Explicit AOT target or ``None`` for the current
            device.
        builder: Callable that creates the _PerCCBuildResults on a cache miss.
            Exactly one thread runs this callable for a given key while other
            threads wait for the result.

    Returns:
        ``(build_results, bound_result)``: the cached or newly built
        _PerCCBuildResults and, for current-device builds, the loaded result
        bound to the constructing device — resolved once here so ``__call__``
        needs no device query. ``bound_result`` is ``None`` for explicit AOT
        builds, which resolve per call.
    """
    from ._cccl_interop import cc_to_key, normalize_compute_capabilities

    if compute_capability is None:
        # The factory decorator already queried the device on the wrapper-cache
        # miss path and hands the result through thread-local state; fall back
        # to a fresh query for direct construction (e.g. deserialization).
        device_info = getattr(_thread_local, "factory_device_info", None)
        if device_info is None:
            device_info = _get_current_device_info()
        device_id, cc = device_info
        packed_cc = cc_to_key(cc)
        # TODO: reduce to _DefaultBuildTarget(packed_cc) once v2 supports
        # build-result serialization.
        target_key = (
            _DefaultBuildTarget(packed_cc)
            if _BACKEND_SERIALIZES_BUILD_RESULTS
            else _DeviceBuildTarget(device_id, cc)
        )
        user_cache_key = _make_cache_key_from_args(*key_args)
        cache_key = (build_result_type, target_key, user_cache_key)
        build_results = _cache_single_flight(
            _process_wide_build_results_cache, cache_key, builder
        )
        return build_results, _bind_default_build(
            build_results, packed_cc, device_id, builder
        )

    target_ccs = normalize_compute_capabilities(compute_capability)
    assert target_ccs is not None
    aot_target_key = _AOTBuildTarget(tuple(cc_to_key(cc) for cc in target_ccs))
    user_cache_key = _make_cache_key_from_args(*key_args)
    aot_cache_key = (build_result_type, aot_target_key, user_cache_key)
    return (
        _cache_single_flight(_process_wide_build_results_cache, aot_cache_key, builder),
        None,
    )


def _bind_default_build(
    build_results, packed_cc: _PackedCCKey, device_id: int, builder
):
    """Resolve the loaded result the constructing device executes.

    Default wrappers are bound to the device that was current at factory-call
    time (their wrapper-cache key includes it), so the binding is resolved
    once here and reused by every ``__call__`` with no device query. For the
    device that built the shared entry this is a warm lookup; another same-cc
    device loads its own clone of the compiled payload here (serialize ->
    deserialize -> load, milliseconds where a build costs a second).

    Cloning is only an optimization and its failures have no stable exception
    type, so any resolve failure falls back to a full build for this device —
    recorded in the shared per-device slot so same-device threads share it —
    and genuine errors surface from the build path itself.
    """
    try:
        return build_results.resolve(packed_cc, device_id)
    except Exception:

        def build_privately():
            (result,) = builder().values()
            return result

        return _cache_single_flight(
            build_results._loaded_results, (packed_cc, device_id), build_privately
        )


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
                target = _DeviceBuildTarget(device_id, cc)
                target_cc_arg = cc
            else:
                target = None
                target_cc_arg = kwargs.get("compute_capability")
            # No thread id in the key: the containing cache is threading.local,
            # so each thread only ever sees its own entries.
            cache_key = (target, user_cache_key)
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
                _thread_local.factory_device_info = target
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

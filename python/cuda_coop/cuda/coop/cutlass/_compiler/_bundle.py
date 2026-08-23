# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve generated CUTLASS providers through the compiler lifecycle.

This module owns resolver ordering, phase telemetry, and orchestration across
target selection, cache, Clang, and NVRTC services. Those services own
their implementation details and artifact state.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable, Hashable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from cutlass._mlir import ir
from cutlass.base_dsl.common import DSLRuntimeError

from cuda.coop._headers import HeaderResolutionError, resolve_include_paths

from . import _cache as _cache_support
from . import _clang as _clang_support
from . import _nvrtc as _nvrtc_support
from ._bundle_contract import (
    RESOLUTION_ROUTE_CLANG,
    RESOLUTION_ROUTE_DISK,
    RESOLUTION_ROUTE_MEMORY,
    RESOLUTION_ROUTE_NVRTC,
    RESOLUTION_ROUTE_PRECOMPILED,
    BundleCompilation,
    BundleResolution,
    BundleResolutionRequest,
    BundleTelemetry,
    LayoutProbe,
    StorageLayout,
    _prepare_layout_probes,
    _validate_storage_layout,
    bundle_compiler_options,
    include_dirs_identity,
    make_bundle_cache_identity,
    make_bundle_identity,
)
from ._target import BUNDLE_FORMAT_ENV

DUMP_DIR_ENV = "CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR"
CCCL_ROOT_ENV = "CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT"
COMMON_CCCL_ROOT_ENV = "CUDA_COOP_CCCL_ROOT"
LINK_LIBRARIES_ATTR = "link-libraries"


_ROUTE_COUNTS: dict[str, int] = {}
_PHASE_COUNTS: dict[str, int] = {}
_PHASE_TIMINGS_NS: dict[str, int] = {}
_UNKNOWN_COMPILER_PROCESS_TOKEN = f"unknown-{os.getpid()}-{os.urandom(8).hex()}"
_ACTIVE_PRECOMPILE_RESOLVERS: ContextVar[
    tuple[Callable[[BundleResolutionRequest], BundleResolution | None], ...]
] = ContextVar(
    "cuda_coop_cutlass_active_precompile_resolvers",
    default=(),
)
_ACTIVE_POST_RESOLUTION_OBSERVERS: ContextVar[
    tuple[Callable[[BundleResolution], None], ...]
] = ContextVar(
    "cuda_coop_cutlass_active_post_resolution_observers",
    default=(),
)


@contextmanager
def activate_bundle_precompile_resolver(
    resolver: Callable[[BundleResolutionRequest], BundleResolution | None],
):
    """Push one context-local resolver, tried before enclosing resolvers."""

    if not callable(resolver):
        raise TypeError("bundle precompile resolver must be callable")
    resolvers = _ACTIVE_PRECOMPILE_RESOLVERS.get()
    token = _ACTIVE_PRECOMPILE_RESOLVERS.set((*resolvers, resolver))
    try:
        yield
    finally:
        _ACTIVE_PRECOMPILE_RESOLVERS.reset(token)


@contextmanager
def activate_bundle_resolution_observer(
    observer: Callable[[BundleResolution], None],
):
    """Push one context-local observer, invoked after enclosing observers."""

    if not callable(observer):
        raise TypeError("bundle resolution observer must be callable")
    observers = _ACTIVE_POST_RESOLUTION_OBSERVERS.get()
    token = _ACTIVE_POST_RESOLUTION_OBSERVERS.set((*observers, observer))
    try:
        yield
    finally:
        _ACTIVE_POST_RESOLUTION_OBSERVERS.reset(token)


def append_link_library_attr(module: ir.Module, path: str) -> None:
    for op in module.body.operations:
        if op.name != "gpu.module":
            continue
        existing: set[str] = set()
        if LINK_LIBRARIES_ATTR in op.attributes:
            existing.update(
                attr.value
                for attr in op.attributes[LINK_LIBRARIES_ATTR]
                if getattr(attr, "value", "")
            )
        existing.add(path)
        op.attributes[LINK_LIBRARIES_ATTR] = ir.ArrayAttr.get(
            [ir.StringAttr.get(x) for x in sorted(existing)]
        )


def maybe_dump_source(source: str, source_hash: str) -> None:
    dump_dir = os.environ.get(DUMP_DIR_ENV)
    if not dump_dir:
        return
    dump_dir = os.path.abspath(os.path.expanduser(dump_dir))
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"cuda_coop_cutlass_bundle_{source_hash}.cpp")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)


def required_cccl_headers(
    source: str,
    *,
    registered_headers: Callable[[], dict[str, str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                relative_path
                for include, relative_path in registered_headers().items()
                if include in source
            }
        )
    )


def cccl_include_dirs(
    required_headers: tuple[str, ...],
    *,
    scope: str,
    provider_dir: str,
) -> list[str]:
    if not required_headers:
        return []

    try:
        paths = resolve_include_paths(
            start=Path(provider_dir),
            configured_roots=(
                os.environ.get(CCCL_ROOT_ENV),
                os.environ.get(COMMON_CCCL_ROOT_ENV),
            ),
            required_headers=required_headers,
        )
    except HeaderResolutionError as exc:
        raise DSLRuntimeError(
            f"{scope} provider could not resolve its CCCL headers.",
            cause=exc,
        ) from exc
    return [str(path) for path in paths.as_tuple()]


def _bundle_compilation(
    cached: _cache_support._CachedBundle,
    key_to_expression: dict[Hashable, str],
) -> BundleCompilation:
    _cache_support.add_managed_bundle_path(cached.path)
    return BundleCompilation(
        path=cached.path,
        layouts={
            key: cached.layouts_by_expression[expression]
            for key, expression in key_to_expression.items()
        },
    )


def _finish_phase(
    phase_timings_ns: dict[str, int],
    phase: str,
    started_ns: int,
) -> None:
    elapsed_ns = max(0, time.perf_counter_ns() - started_ns)
    phase_timings_ns[phase] = phase_timings_ns.get(phase, 0) + elapsed_ns


def _validated_precompiled_bundle(
    resolution: BundleResolution,
    request: BundleResolutionRequest,
) -> _cache_support._CachedBundle:
    if not isinstance(resolution, BundleResolution):
        raise TypeError(
            "bundle precompile resolver must return BundleResolution or None"
        )
    if resolution.request != request:
        raise ValueError("precompiled bundle resolution does not match its request")
    if resolution.route != RESOLUTION_ROUTE_PRECOMPILED:
        raise ValueError("precompiled bundle resolution has an invalid route")
    if not isinstance(resolution.path, str) or not resolution.path:
        raise ValueError("precompiled bundle resolution requires a non-empty path")
    if not os.path.isfile(resolution.path):
        raise ValueError("precompiled bundle resolution path is not a file")
    layout_expressions = request.identity.layout_expressions
    if set(resolution.layouts_by_expression) != set(layout_expressions):
        raise ValueError(
            "precompiled bundle resolution has incompatible layout metadata"
        )
    layouts_by_expression = {}
    for expression in layout_expressions:
        layout = resolution.layouts_by_expression[expression]
        if not isinstance(layout, StorageLayout):
            raise TypeError(
                "precompiled bundle layout metadata must contain StorageLayout values"
            )
        layouts_by_expression[expression] = _validate_storage_layout(
            layout.size_in_bytes,
            layout.alignment,
            description=expression,
        )
    return _cache_support._CachedBundle(
        path=resolution.path,
        layouts_by_expression=layouts_by_expression,
        producer_compiler=resolution.producer_compiler,
        producer_compiler_version=resolution.producer_compiler_version,
        producer_toolkit_version=resolution.producer_toolkit_version,
    )


def _finish_bundle_resolution(
    *,
    request: BundleResolutionRequest,
    cached: _cache_support._CachedBundle,
    route: str,
    key_to_expression: dict[Hashable, str],
    phase_timings_ns: dict[str, int],
    resolution_started_ns: int,
) -> BundleCompilation:
    _finish_phase(phase_timings_ns, "total", resolution_started_ns)
    resolution = BundleResolution(
        request=request,
        path=cached.path,
        layouts_by_expression=dict(cached.layouts_by_expression),
        route=route,
        producer_compiler=cached.producer_compiler,
        producer_compiler_version=cached.producer_compiler_version,
        producer_toolkit_version=cached.producer_toolkit_version,
        phase_timings_ns=dict(phase_timings_ns),
    )
    with _cache_support._STATE_LOCK:
        _ROUTE_COUNTS[route] = _ROUTE_COUNTS.get(route, 0) + 1
        for phase, duration_ns in phase_timings_ns.items():
            _PHASE_COUNTS[phase] = _PHASE_COUNTS.get(phase, 0) + 1
            _PHASE_TIMINGS_NS[phase] = _PHASE_TIMINGS_NS.get(phase, 0) + duration_ns
    for observer in _ACTIVE_POST_RESOLUTION_OBSERVERS.get():
        observer(resolution)
    return _bundle_compilation(cached, key_to_expression)


def _compile_bundle_source(
    source: str,
    *,
    layout_probes: Iterable[LayoutProbe],
    scope: str,
    provider_dir: str,
    registered_headers: Callable[[], dict[str, str]],
    select_bundle_format: Callable[[], str],
    resolve_nvrtc_sm_arch: Callable[[], str],
    resolve_nvrtc_arch: Callable[[], str],
    which: Callable[[str], str | None] = shutil.which,
) -> BundleCompilation:
    resolution_started_ns = time.perf_counter_ns()
    phase_timings_ns: dict[str, int] = {}
    phase_started_ns = time.perf_counter_ns()
    prepared_probes = _prepare_layout_probes(source, layout_probes)
    source = prepared_probes.source
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    _finish_phase(phase_timings_ns, "prepare", phase_started_ns)

    phase_started_ns = time.perf_counter_ns()
    bundle_format = select_bundle_format()
    if prepared_probes.expressions and bundle_format != "ltoir":
        raise DSLRuntimeError(
            f"{scope} provider layout metadata requires an NVRTC LTO-IR bundle; "
            f"set {BUNDLE_FORMAT_ENV}=ltoir."
        )
    bundle_sm_arch = "none"
    if bundle_format == "bc":
        bundle_arch = "nvptx64"
    else:
        bundle_sm_arch = resolve_nvrtc_sm_arch()
        bundle_arch = resolve_nvrtc_arch()
    _finish_phase(phase_timings_ns, "target", phase_started_ns)

    compiler_options = bundle_compiler_options(bundle_format, bundle_arch)
    identity = make_bundle_identity(
        source_hash=source_hash,
        bundle_format=bundle_format,
        bundle_arch=bundle_arch,
        bundle_sm_arch=bundle_sm_arch,
        compiler_options=compiler_options,
        layout_expressions=prepared_probes.expressions,
    )
    request = BundleResolutionRequest(
        identity=identity,
        source=source,
    )

    resolvers = _ACTIVE_PRECOMPILE_RESOLVERS.get()
    if resolvers:
        phase_started_ns = time.perf_counter_ns()
        for resolver in reversed(resolvers):
            resolution = resolver(request)
            if resolution is None:
                continue
            _finish_phase(phase_timings_ns, "precompile_resolvers", phase_started_ns)
            return _finish_bundle_resolution(
                request=request,
                cached=_validated_precompiled_bundle(resolution, request),
                route=RESOLUTION_ROUTE_PRECOMPILED,
                key_to_expression=prepared_probes.key_to_expression,
                phase_timings_ns=phase_timings_ns,
                resolution_started_ns=resolution_started_ns,
            )
        _finish_phase(phase_timings_ns, "precompile_resolvers", phase_started_ns)

    phase_started_ns = time.perf_counter_ns()
    include_dirs = cccl_include_dirs(
        required_cccl_headers(source, registered_headers=registered_headers),
        scope=scope,
        provider_dir=provider_dir,
    )
    _finish_phase(phase_timings_ns, "header_resolution", phase_started_ns)
    phase_started_ns = time.perf_counter_ns()
    include_identity = include_dirs_identity(include_dirs)
    include_key = include_identity.digest
    _finish_phase(phase_timings_ns, "header_fingerprint", phase_started_ns)
    for root_identity in include_identity.roots:
        phase = f"header_identity_{root_identity.method.replace('-', '_')}"
        phase_timings_ns[phase] = (
            phase_timings_ns.get(phase, 0) + root_identity.duration_ns
        )
    clangxx: str | None = None
    clang_version: str | None = None
    if bundle_format == "ltoir":
        nvrtc_version = _nvrtc_support.get_version()
        cache_compiler_version = (
            nvrtc_version
            if nvrtc_version is not None
            else _UNKNOWN_COMPILER_PROCESS_TOKEN
        )
    else:
        nvrtc_version = None
        clangxx, clang_version, cache_compiler_version = (
            _clang_support.resolve_clang_compiler(which)
        )
    cache_identity = make_bundle_cache_identity(
        identity,
        include_key=include_key,
        producer_compiler_version=cache_compiler_version,
    )

    # Source inspection is independent of JIT compilation. Keep the dump useful
    # even when this process or a previous process already populated the cache.
    phase_started_ns = time.perf_counter_ns()
    maybe_dump_source(source, source_hash)
    _finish_phase(phase_timings_ns, "source_dump", phase_started_ns)

    phase_started_ns = time.perf_counter_ns()
    cached = _cache_support.memory_cached_bundle(cache_identity.cache_key)
    memory_hit = cached is not None
    _finish_phase(phase_timings_ns, "memory_cache", phase_started_ns)
    if memory_hit:
        assert cached is not None
        return _finish_bundle_resolution(
            request=request,
            cached=cached,
            route=RESOLUTION_ROUTE_MEMORY,
            key_to_expression=prepared_probes.key_to_expression,
            phase_timings_ns=phase_timings_ns,
            resolution_started_ns=resolution_started_ns,
        )

    phase_started_ns = time.perf_counter_ns()
    cache_dir = _cache_support.ensure_cache_dir(scope)
    output_path = os.path.join(
        cache_dir,
        f"{cache_identity.artifact_stem}.{bundle_format}",
    )
    lock_started_ns = time.perf_counter_ns()
    with _cache_support.artifact_lock(output_path, scope=scope):
        _finish_phase(phase_timings_ns, "artifact_lock", lock_started_ns)

        # A thread or process may have populated the artifact while this caller
        # waited for the per-artifact lock.
        phase_started_ns = time.perf_counter_ns()
        cached = _cache_support.memory_cached_bundle(cache_identity.cache_key)
        memory_hit = cached is not None
        _finish_phase(phase_timings_ns, "memory_cache_after_lock", phase_started_ns)
        if memory_hit:
            assert cached is not None
            return _finish_bundle_resolution(
                request=request,
                cached=cached,
                route=RESOLUTION_ROUTE_MEMORY,
                key_to_expression=prepared_probes.key_to_expression,
                phase_timings_ns=phase_timings_ns,
                resolution_started_ns=resolution_started_ns,
            )

        phase_started_ns = time.perf_counter_ns()
        cached = None
        if os.path.exists(output_path):
            cached = _cache_support._load_bundle_metadata(
                output_path,
                prepared_probes.expressions,
                cache_identity,
            )
        _finish_phase(phase_timings_ns, "disk_cache", phase_started_ns)
        if cached is not None:
            _cache_support.store_memory_bundle(cache_identity.cache_key, cached)
            return _finish_bundle_resolution(
                request=request,
                cached=cached,
                route=RESOLUTION_ROUTE_DISK,
                key_to_expression=prepared_probes.key_to_expression,
                phase_timings_ns=phase_timings_ns,
                resolution_started_ns=resolution_started_ns,
            )

        phase_started_ns = time.perf_counter_ns()
        if bundle_format == "bc":
            cached = _clang_support.compile_bundle(
                source,
                output_path=output_path,
                cache_dir=cache_dir,
                cache_identity=cache_identity,
                compiler_options=compiler_options,
                include_dirs=include_dirs,
                scope=scope,
                clangxx=clangxx,
                clang_version=clang_version,
            )
            route = RESOLUTION_ROUTE_CLANG
        else:
            cached = _nvrtc_support.compile_bundle(
                source,
                output_path=output_path,
                cache_identity=cache_identity,
                compiler_options=compiler_options,
                include_dirs=include_dirs,
                prepared_probes=prepared_probes,
                nvrtc_version=nvrtc_version,
                scope=scope,
            )
            route = RESOLUTION_ROUTE_NVRTC
        _cache_support.store_memory_bundle(cache_identity.cache_key, cached)
        _cache_support.record_compilation()
    _finish_phase(phase_timings_ns, "compiler", phase_started_ns)
    return _finish_bundle_resolution(
        request=request,
        cached=cached,
        route=route,
        key_to_expression=prepared_probes.key_to_expression,
        phase_timings_ns=phase_timings_ns,
        resolution_started_ns=resolution_started_ns,
    )


def compile_bundle_source(
    source: str,
    *,
    scope: str,
    provider_dir: str,
    registered_headers: Callable[[], dict[str, str]],
    select_bundle_format: Callable[[], str],
    resolve_nvrtc_sm_arch: Callable[[], str],
    resolve_nvrtc_arch: Callable[[], str],
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """Compile a provider bundle and return its linkable artifact path."""

    return _compile_bundle_source(
        source,
        layout_probes=(),
        scope=scope,
        provider_dir=provider_dir,
        registered_headers=registered_headers,
        select_bundle_format=select_bundle_format,
        resolve_nvrtc_sm_arch=resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=resolve_nvrtc_arch,
        which=which,
    ).path


def compile_bundle_source_with_layouts(
    source: str,
    *,
    layout_probes: Iterable[LayoutProbe],
    scope: str,
    provider_dir: str,
    registered_headers: Callable[[], dict[str, str]],
    select_bundle_format: Callable[[], str],
    resolve_nvrtc_sm_arch: Callable[[], str],
    resolve_nvrtc_arch: Callable[[], str],
    which: Callable[[str], str | None] = shutil.which,
) -> BundleCompilation:
    """Compile one LTO-IR bundle and recover exact layouts from that program."""

    return _compile_bundle_source(
        source,
        layout_probes=layout_probes,
        scope=scope,
        provider_dir=provider_dir,
        registered_headers=registered_headers,
        select_bundle_format=select_bundle_format,
        resolve_nvrtc_sm_arch=resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=resolve_nvrtc_arch,
        which=which,
    )


def reset_compile_state() -> None:
    """Reset orchestration, cache, and NVRTC counters together."""

    with _cache_support._STATE_LOCK:
        _cache_support.reset_compile_state()
        _nvrtc_support.reset_compile_state()
        _ROUTE_COUNTS.clear()
        _PHASE_COUNTS.clear()
        _PHASE_TIMINGS_NS.clear()


def get_compile_counter() -> int:
    return _cache_support.get_compile_counter()


def get_nvrtc_compile_program_counter() -> int:
    return _nvrtc_support.get_compile_program_counter()


def managed_bundle_paths() -> frozenset[str]:
    return _cache_support.managed_bundle_paths()


def get_bundle_telemetry() -> BundleTelemetry:
    with _cache_support._STATE_LOCK:
        return BundleTelemetry(
            route_counts=dict(_ROUTE_COUNTS),
            phase_counts=dict(_PHASE_COUNTS),
            phase_timings_ns=dict(_PHASE_TIMINGS_NS),
        )

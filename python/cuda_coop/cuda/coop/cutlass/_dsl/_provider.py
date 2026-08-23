# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared provider metadata for CuTe cooperative CUB shims."""

from __future__ import annotations

import importlib
import math
import re
import threading
import weakref
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

import numpy as np
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.typing import (
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
)

from cuda.coop._core.dtype_policy import validate_common_v1_numeric_dtype_name
from cuda.coop._core.root_api import _common_root_operation_name

from .._value_metadata import (
    ValueGroupMetadata,
    register_scalar_metadata_lookup,
)
from ._thread_data import ThreadData


@dataclass(frozen=True)
class TypeSpec:
    cpp_type: str
    token: str
    width_bits: int
    zero_literal: str
    shfl_up: str
    shfl_idx: str
    smem_store: str
    smem_load: str


@dataclass(frozen=True)
class BundleRenderer:
    include_lines: tuple[str, ...]
    cccl_headers: tuple[tuple[str, str], ...]
    render: Callable[[Any], list[str]]
    scratch_layout_probe: Callable[[Any], "ScratchLayoutProbe | None"] | None = None


@dataclass(frozen=True)
class ScratchLayout:
    """Exact C++ temporary-storage layout for one specialization."""

    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class ScratchLayoutProbe:
    """C++ constant expressions for one exact scratch layout."""

    requirement_key: Hashable
    size_expression: str
    alignment_expression: str


@dataclass(frozen=True)
class DeferredTempStorageEvent:
    """One traced cooperative call whose scratch operands need finalization."""

    kernel_op: Any
    kernel_name: str
    temp_storage: Any
    primitive_name: str
    requirement_key: Hashable
    sharing: str
    auto_sync: bool
    capacity_size_in_bytes: int | None
    capacity_alignment: int | None
    smem_addr_placeholder: Any
    size_placeholder: Any
    location: str


@dataclass(frozen=True)
class DeferredTempStorageBinding:
    """Resolved per-call scratch slice within a deferred storage plan."""

    event: DeferredTempStorageEvent
    byte_offset_in_bytes: int
    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class DeferredTempStoragePlan:
    """One kernel-local allocation for one TempStorage identity."""

    kernel_op: Any
    kernel_name: str
    temp_storage: Any
    size_in_bytes: int
    alignment: int
    bindings: tuple[DeferredTempStorageBinding, ...]


@dataclass(frozen=True)
class _ScalarResultTypeEntry:
    value_type: type
    value_ref: weakref.ReferenceType[Any] | None
    strong_ref: Any | None = None
    group_metadata: ValueGroupMetadata | None = None


TYPE_SPECS: dict[type, TypeSpec] = {
    Uint8: TypeSpec(
        cpp_type="unsigned char",
        token="u8",
        width_bits=8,
        zero_literal="0u",
        shfl_up="cuda_coop_cutlass_shfl_up_u8",
        shfl_idx="cuda_coop_cutlass_shfl_idx_u8",
        smem_store="cuda_coop_cutlass_st_shared_u8",
        smem_load="cuda_coop_cutlass_ld_shared_u8",
    ),
    Int32: TypeSpec(
        cpp_type="int",
        token="i32",
        width_bits=32,
        zero_literal="0",
        shfl_up="cuda_coop_cutlass_shfl_up_i32",
        shfl_idx="cuda_coop_cutlass_shfl_idx_i32",
        smem_store="cuda_coop_cutlass_st_shared_i32",
        smem_load="cuda_coop_cutlass_ld_shared_i32",
    ),
    Uint32: TypeSpec(
        cpp_type="unsigned int",
        token="u32",
        width_bits=32,
        zero_literal="0u",
        shfl_up="cuda_coop_cutlass_shfl_up_u32",
        shfl_idx="cuda_coop_cutlass_shfl_idx_u32",
        smem_store="cuda_coop_cutlass_st_shared_u32",
        smem_load="cuda_coop_cutlass_ld_shared_u32",
    ),
    Int64: TypeSpec(
        cpp_type="long long",
        token="i64",
        width_bits=64,
        zero_literal="0ll",
        shfl_up="cuda_coop_cutlass_shfl_up_i64",
        shfl_idx="cuda_coop_cutlass_shfl_idx_i64",
        smem_store="cuda_coop_cutlass_st_shared_i64",
        smem_load="cuda_coop_cutlass_ld_shared_i64",
    ),
    Uint64: TypeSpec(
        cpp_type="unsigned long long",
        token="u64",
        width_bits=64,
        zero_literal="0ull",
        shfl_up="cuda_coop_cutlass_shfl_up_u64",
        shfl_idx="cuda_coop_cutlass_shfl_idx_u64",
        smem_store="cuda_coop_cutlass_st_shared_u64",
        smem_load="cuda_coop_cutlass_ld_shared_u64",
    ),
    Float32: TypeSpec(
        cpp_type="float",
        token="f32",
        width_bits=32,
        zero_literal="0.0f",
        shfl_up="cuda_coop_cutlass_shfl_up_f32",
        shfl_idx="cuda_coop_cutlass_shfl_idx_f32",
        smem_store="cuda_coop_cutlass_st_shared_f32",
        smem_load="cuda_coop_cutlass_ld_shared_f32",
    ),
    Float64: TypeSpec(
        cpp_type="double",
        token="f64",
        width_bits=64,
        zero_literal="0.0",
        shfl_up="cuda_coop_cutlass_shfl_up_f64",
        shfl_idx="cuda_coop_cutlass_shfl_idx_f64",
        smem_store="cuda_coop_cutlass_st_shared_f64",
        smem_load="cuda_coop_cutlass_ld_shared_f64",
    ),
}

SCAN_REDUCE_TYPES = frozenset(TYPE_SPECS.keys())
RADIX_KEY_TYPES = frozenset({Int32, Uint32, Int64, Uint64})
ALL_PROVIDER_TYPES = frozenset(TYPE_SPECS.keys())
ORDINARY_PROVIDER_TYPES = {
    int: Int32,
    float: Float32,
    np.uint8: Uint8,
    np.int32: Int32,
    np.uint32: Uint32,
    np.int64: Int64,
    np.uint64: Uint64,
    np.float32: Float32,
    np.float64: Float64,
}
PROVIDER_TYPE_NAMES = {
    Uint8: "uint8",
    Int32: "int32",
    Uint32: "uint32",
    Int64: "int64",
    Uint64: "uint64",
    Float32: "float32",
    Float64: "float64",
}
_INTEGER_TYPE_TOKENS = frozenset({"u8", "i32", "u32", "i64", "u64"})
_FLOAT_TYPE_TOKENS = frozenset({"f32", "f64"})
_NOT_PLAIN_SCALAR = object()
_SESSION_SCOPE = "cuda.coop.cutlass"
_TRACE_HOOK_DISPATCHER_ATTR = "_cuda_coop_cutlass_provider_trace_finalize_dispatcher"
_TRACE_HOOK_TARGET_ATTR = "_cuda_coop_cutlass_provider_trace_finalize_hook"
_BUILTIN_SCALAR_VALUE_TYPES = (bool, int, float, complex, str)
_FEATURE_DEFINE_RE = re.compile(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s|\(|$)")
_BUNDLE_RENDERERS: dict[str, BundleRenderer] = {}
_BUNDLE_FINALIZER: Callable[[Any, Any, str], None] | None = None
_COMPILE_OPTIONS_UNSET = object()
_STATE_LOCK = threading.RLock()
_SCALAR_RESULT_TYPES: dict[tuple[int, type], _ScalarResultTypeEntry] = {}
_SESSIONS: weakref.WeakKeyDictionary[Any, "BundleSession"] = weakref.WeakKeyDictionary()
_ID_SESSIONS: dict[int, tuple[weakref.ReferenceType[Any], "BundleSession", Any]] = {}


def supported_names(types: frozenset[type]) -> str:
    return "/".join(sorted(t.__name__ for t in types))


def coerce_plain_scalar(
    value: Any,
    value_type: type,
    *,
    name: str,
    scope: str,
    allow_nonfinite: bool,
    convert: bool = True,
) -> Any:
    """Validate and optionally convert an exact Python numeric literal."""

    token = TYPE_SPECS[value_type].token
    if type(value) is int:
        if token not in _INTEGER_TYPE_TOKENS:
            raise TypeError(
                f"{scope}.{name} dtype does not match {value_type.__name__}"
            )
        bits = int(token.lstrip("iu"))
        lower = 0 if token.startswith("u") else -(1 << (bits - 1))
        upper = (1 << bits) - 1 if token.startswith("u") else (1 << (bits - 1)) - 1
        if not lower <= value <= upper:
            raise ValueError(
                f"{scope}.{name}={value} is not representable in {value_type.__name__}"
            )
        return value_type(value) if convert else value
    if type(value) is float:
        if token not in _FLOAT_TYPE_TOKENS:
            raise TypeError(
                f"{scope}.{name} dtype does not match {value_type.__name__}"
            )
        if not allow_nonfinite and not math.isfinite(value):
            raise ValueError(f"{scope}.{name} must be finite")
        numpy_type = np.float32 if token == "f32" else np.float64
        limit = float(np.finfo(numpy_type).max)
        if math.isfinite(value) and abs(value) > limit:
            raise ValueError(
                f"{scope}.{name}={value} is not representable in {value_type.__name__}"
            )
        return value_type(value) if convert else value
    return _NOT_PLAIN_SCALAR


def register_bundle_renderer(
    kind: str,
    *,
    render: Callable[[Any], list[str]],
    include_lines: tuple[str, ...] = (),
    cccl_headers: tuple[tuple[str, str], ...] = (),
    scratch_layout_probe: Callable[[Any], ScratchLayoutProbe | None] | None = None,
) -> None:
    """Register an internal non-block request renderer with the shared bundle JIT."""
    if not kind:
        raise ValueError("kind must be a non-empty string")
    if not callable(render):
        raise TypeError("render must be callable")
    if scratch_layout_probe is not None and not callable(scratch_layout_probe):
        raise TypeError("scratch_layout_probe must be callable or None")
    _BUNDLE_RENDERERS[kind] = BundleRenderer(
        include_lines=tuple(include_lines),
        cccl_headers=tuple(cccl_headers),
        render=render,
        scratch_layout_probe=scratch_layout_probe,
    )


def bundle_renderer_for(request: Any) -> BundleRenderer | None:
    return _BUNDLE_RENDERERS.get(getattr(request, "kind", ""))


def canonical_bundle_requests(requests: Iterable[Any]) -> tuple[Any, ...]:
    """Return one deterministic request per provider symbol."""

    requests_by_symbol: dict[str, Any] = {}
    for request in requests:
        symbol = getattr(request, "symbol_name", None)
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("provider bundle requests require a non-empty symbol_name")
        existing = requests_by_symbol.get(symbol)
        if existing is not None and existing != request:
            raise ValueError(
                f"provider symbol {symbol!r} maps to conflicting bundle requests"
            )
        requests_by_symbol[symbol] = request
    return tuple(requests_by_symbol[symbol] for symbol in sorted(requests_by_symbol))


def canonical_bundle_preamble_lines(lines: Iterable[str]) -> tuple[str, ...]:
    """Canonicalize feature definitions before all other preamble lines."""

    feature_definitions: dict[str, str] = {}
    other_lines: set[str] = set()
    for line in lines:
        if not line:
            continue
        if line.startswith("#define "):
            match = _FEATURE_DEFINE_RE.match(line)
            if match is None:
                raise ValueError(f"invalid provider feature definition: {line!r}")
            name = match.group(1)
            existing = feature_definitions.get(name)
            if existing is not None and existing != line:
                raise ValueError(
                    f"provider feature {name!r} has conflicting definitions"
                )
            feature_definitions[name] = line
        else:
            other_lines.add(line)
    return (
        *(feature_definitions[name] for name in sorted(feature_definitions)),
        *sorted(other_lines),
    )


def bundle_include_lines(requests: Iterable[Any]) -> list[str]:
    include_lines: list[str] = []
    for request in canonical_bundle_requests(requests):
        renderer = bundle_renderer_for(request)
        if renderer is not None:
            include_lines.extend(renderer.include_lines)
    return list(canonical_bundle_preamble_lines(include_lines))


def registered_bundle_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    for kind in sorted(_BUNDLE_RENDERERS):
        renderer = _BUNDLE_RENDERERS[kind]
        for include, relative_path in sorted(renderer.cccl_headers):
            existing = headers.get(include)
            if existing is not None and existing != relative_path:
                raise ValueError(
                    f"provider include {include!r} maps to conflicting CCCL headers"
                )
            headers[include] = relative_path
    return {include: headers[include] for include in sorted(headers)}


def bundle_scratch_layout_probes(
    requests: list[Any],
) -> dict[Hashable, ScratchLayoutProbe]:
    """Collect exact-layout probes registered by provider request renderers."""

    probes: dict[Hashable, ScratchLayoutProbe] = {}
    for request in canonical_bundle_requests(requests):
        renderer = bundle_renderer_for(request)
        if renderer is None or renderer.scratch_layout_probe is None:
            continue
        probe = renderer.scratch_layout_probe(request)
        if probe is None:
            continue
        try:
            hash(probe.requirement_key)
        except TypeError as exc:
            raise TypeError("scratch requirement keys must be hashable") from exc
        existing = probes.get(probe.requirement_key)
        if existing is not None and existing != probe:
            raise ValueError(
                "scratch requirement key maps to conflicting NVRTC layout probes"
            )
        probes[probe.requirement_key] = probe
    return probes


def make_scratch_layout_probe(
    requirement_key: Hashable,
    cpp_type: str,
) -> ScratchLayoutProbe:
    """Describe a name expression for the exact layout of ``cpp_type``."""

    if not cpp_type:
        raise ValueError("scratch layout probe requires a non-empty C++ type")
    return ScratchLayoutProbe(
        requirement_key=requirement_key,
        size_expression=f"sizeof({cpp_type})",
        alignment_expression=f"alignof({cpp_type})",
    )


class BundleSession:
    def __init__(self, trace_module_op: Any | None = None) -> None:
        self._lock = threading.RLock()
        self.trace_module_op = trace_module_op
        self.requests: set[Any] = set()
        self.temp_storage_bindings: weakref.WeakKeyDictionary[Any, Any] = (
            weakref.WeakKeyDictionary()
        )
        self.scalar_result_type_entries: dict[
            tuple[int, type], _ScalarResultTypeEntry
        ] = {}
        self.deferred_temp_storage_events: list[DeferredTempStorageEvent] = []

    def close(self) -> None:
        try:
            with _STATE_LOCK:
                _clear_scalar_result_types_for_session_locked(self)
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def add(self, request: Any) -> None:
        with self._lock:
            self.requests.add(request)

    def add_scalar_result_type_entry(
        self,
        key: tuple[int, type],
        entry: _ScalarResultTypeEntry,
    ) -> None:
        with self._lock:
            self.scalar_result_type_entries[key] = entry

    def add_deferred_temp_storage_event(
        self,
        event: DeferredTempStorageEvent,
    ) -> None:
        with self._lock:
            self.deferred_temp_storage_events.append(event)

    def is_empty(self) -> bool:
        with self._lock:
            return not self.requests and not self.deferred_temp_storage_events

    def snapshot(self):
        with self._lock:
            return (
                set(self.requests),
                dict(self.temp_storage_bindings),
                dict(self.scalar_result_type_entries),
                list(self.deferred_temp_storage_events),
            )

    def restore(self, snapshot) -> None:
        if len(snapshot) == 3:
            requests, temp_storage_bindings, scalar_result_type_entries = snapshot
            deferred_temp_storage_events = []
        else:
            (
                requests,
                temp_storage_bindings,
                scalar_result_type_entries,
                deferred_temp_storage_events,
            ) = snapshot
        restored_entries = dict(scalar_result_type_entries)
        with _STATE_LOCK:
            with self._lock:
                for key, entry in self.scalar_result_type_entries.items():
                    if key in restored_entries:
                        continue
                    if _SCALAR_RESULT_TYPES.get(key) is entry:
                        _SCALAR_RESULT_TYPES.pop(key, None)
                self.requests = set(requests)
                self.temp_storage_bindings = weakref.WeakKeyDictionary(
                    temp_storage_bindings
                )
                self.scalar_result_type_entries = restored_entries
                self.deferred_temp_storage_events = list(deferred_temp_storage_events)

    def get_temp_storage_binding(self, temp_storage: Any) -> Any | None:
        with self._lock:
            return self.temp_storage_bindings.get(temp_storage)

    def set_temp_storage_binding(self, temp_storage: Any, binding: Any) -> None:
        with self._lock:
            self.temp_storage_bindings[temp_storage] = binding

    def request_list(self) -> list[Any]:
        with self._lock:
            return list(canonical_bundle_requests(self.requests))

    def deferred_temp_storage_event_list(self) -> list[DeferredTempStorageEvent]:
        with self._lock:
            return list(self.deferred_temp_storage_events)

    def belongs_to_trace_module(self, trace_module_op: Any) -> bool:
        with self._lock:
            return self.trace_module_op is None or _same_mlir_operation(
                self.trace_module_op,
                trace_module_op,
            )

    def bind_trace_module(self, trace_module_op: Any) -> bool:
        with self._lock:
            if self.trace_module_op is None:
                self.trace_module_op = trace_module_op
                return True
            return _same_mlir_operation(self.trace_module_op, trace_module_op)

    def scalar_result_type_entry_items(
        self,
    ) -> list[tuple[tuple[int, type], _ScalarResultTypeEntry]]:
        with self._lock:
            return list(self.scalar_result_type_entries.items())

    def scalar_result_type_key_list(self) -> list[tuple[int, type]]:
        with self._lock:
            return list(self.scalar_result_type_entries)

    def clear_scalar_result_type_keys(self) -> None:
        with self._lock:
            self.scalar_result_type_entries.clear()


def register_bundle_finalizer(
    finalizer: Callable[[Any, Any, str], None],
    *,
    scope: str = _SESSION_SCOPE,
) -> None:
    if not callable(finalizer):
        raise TypeError("finalizer must be callable")
    global _BUNDLE_FINALIZER, _SESSION_SCOPE
    with _STATE_LOCK:
        _BUNDLE_FINALIZER = finalizer
        _SESSION_SCOPE = scope


def _ensure_bundle_finalizer() -> Callable[[Any, Any, str], None]:
    if _BUNDLE_FINALIZER is None:
        importlib.import_module(f"{__package__}.block._provider")
    if _BUNDLE_FINALIZER is None:
        raise DSLRuntimeError(
            f"{_SESSION_SCOPE} provider has no cooperative bundle finalizer."
        )
    return _BUNDLE_FINALIZER


def _get_cute_dsl():
    from cutlass.cute import _dsl as cute_dsl

    return cute_dsl.CuTeDSL._get_dsl()


def _trace_finalize_dispatcher(dsl, module, function_name) -> None:
    hook = getattr(dsl, _TRACE_HOOK_TARGET_ATTR, None)
    if hook is not None:
        hook(dsl, module, function_name)


def ensure_trace_hook_registered(
    *,
    finalizer: Callable[[Any, Any, str], None] | None = None,
    scope: str | None = None,
    get_cute_dsl: Callable[[], Any] | None = None,
) -> None:
    if finalizer is None:
        finalizer = _ensure_bundle_finalizer()
        scope = _SESSION_SCOPE if scope is None else scope
    else:
        scope = _SESSION_SCOPE if scope is None else scope
        with _STATE_LOCK:
            needs_registration = (
                _BUNDLE_FINALIZER is not finalizer or _SESSION_SCOPE != scope
            )
        if needs_registration:
            register_bundle_finalizer(finalizer, scope=scope)

    dsl = _get_cute_dsl() if get_cute_dsl is None else get_cute_dsl()
    if getattr(dsl, _TRACE_HOOK_DISPATCHER_ATTR, None) is None:
        register_hook = getattr(dsl, "register_trace_finalize_hook", None)
        if register_hook is None:
            raise DSLRuntimeError(
                f"{scope} provider requires CuTe DSL trace-finalize hook "
                "support so generated cooperative-primitive bundles can be linked "
                "into the compiled kernel. Install a compatible CUTLASS DSL "
                "runtime separately: a nvidia-cutlass-dsl release with "
                "register_trace_finalize_hook and link-libraries support."
            )
        register_hook(_trace_finalize_dispatcher)
        setattr(
            dsl,
            _TRACE_HOOK_DISPATCHER_ATTR,
            _trace_finalize_dispatcher,
        )
    setattr(dsl, _TRACE_HOOK_TARGET_ATTR, finalizer)
    dsl._cuda_coop_cutlass_provider_trace_hook_registered = True


def _ensure_trace_hook_registered() -> None:
    ensure_trace_hook_registered()


def lookup_bundle_session(compile_options: Any) -> BundleSession | None:
    with _STATE_LOCK:
        try:
            return _SESSIONS.get(compile_options)
        except TypeError:
            key = id(compile_options)
            entry = _ID_SESSIONS.get(key)
            if entry is None:
                return None
            compile_options_ref, session, _finalizer = entry
            if compile_options_ref() is compile_options:
                return session
            _ID_SESSIONS.pop(key, None)
            return None


def _drop_id_session(key: int) -> None:
    with _STATE_LOCK:
        entry = _ID_SESSIONS.pop(key, None)
        if entry is not None:
            entry[1].close()


def _clear_scalar_result_types_for_session_locked(session: BundleSession) -> None:
    for key, entry in session.scalar_result_type_entry_items():
        if _SCALAR_RESULT_TYPES.get(key) is entry:
            _SCALAR_RESULT_TYPES.pop(key, None)
    session.clear_scalar_result_type_keys()


def set_bundle_session(compile_options: Any, session: BundleSession) -> None:
    with _STATE_LOCK:
        try:
            _SESSIONS[compile_options] = session
        except TypeError:
            try:
                compile_options_ref = weakref.ref(compile_options)
            except TypeError as exc:
                raise DSLRuntimeError(
                    f"{_SESSION_SCOPE} provider compile_options must be "
                    "hashable or weak-referenceable."
                ) from exc
            key = id(compile_options)
            finalizer = weakref.finalize(compile_options, _drop_id_session, key)
            _ID_SESSIONS[key] = (compile_options_ref, session, finalizer)


def pop_bundle_session(compile_options: Any) -> BundleSession | None:
    with _STATE_LOCK:
        try:
            session = _SESSIONS.pop(compile_options, None)
            if session is not None:
                session.close()
            return session
        except TypeError:
            entry = _ID_SESSIONS.pop(id(compile_options), None)
            if entry is None:
                return None
            compile_options_ref, session, finalizer = entry
            if finalizer.alive:
                finalizer.detach()
            if compile_options_ref() is compile_options:
                session.close()
                return session
            return None


def _same_mlir_operation(lhs: Any, rhs: Any) -> bool:
    lhs = getattr(lhs, "operation", lhs)
    rhs = getattr(rhs, "operation", rhs)
    if lhs is rhs:
        return True
    try:
        result = lhs == rhs
    except Exception:
        return False
    return isinstance(result, bool) and result


def _active_trace_module_op() -> Any | None:
    try:
        from cutlass._mlir import ir

        current_ip = ir.InsertionPoint.current
        op = None if current_ip is None else current_ip.block.owner
    except Exception:
        return None

    while op is not None:
        operation = getattr(op, "operation", op)
        if str(getattr(operation, "name", "")) == "builtin.module":
            return operation
        op = getattr(operation, "parent", None)
    return None


def get_or_create_bundle_session(
    compile_options: Any,
    *,
    trace_module_op: Any | None = None,
) -> BundleSession:
    with _STATE_LOCK:
        session = lookup_bundle_session(compile_options)
        if session is not None and trace_module_op is not None:
            if not session.bind_trace_module(trace_module_op):
                pop_bundle_session(compile_options)
                session = None
        if session is None:
            session = BundleSession(trace_module_op=trace_module_op)
            set_bundle_session(compile_options, session)
        return session


def active_bundle_session() -> BundleSession:
    _ensure_trace_hook_registered()
    compile_options = _get_cute_dsl().compile_options
    return get_or_create_bundle_session(
        compile_options,
        trace_module_op=_active_trace_module_op(),
    )


def active_bundle_session_for(
    *,
    get_cute_dsl: Callable[[], Any],
    ensure_trace_hook: Callable[[], None],
) -> BundleSession:
    ensure_trace_hook()
    compile_options = get_cute_dsl().compile_options
    return get_or_create_bundle_session(
        compile_options,
        trace_module_op=_active_trace_module_op(),
    )


def snapshot_active_session_state_for(*, get_cute_dsl: Callable[[], Any]):
    compile_options = get_cute_dsl().compile_options
    session = lookup_bundle_session(compile_options)
    trace_module_op = _active_trace_module_op()
    if (
        session is not None
        and trace_module_op is not None
        and not session.bind_trace_module(trace_module_op)
    ):
        pop_bundle_session(compile_options)
        session = None
    if session is None:
        return compile_options, None
    return compile_options, session.snapshot()


def snapshot_active_session_state():
    return snapshot_active_session_state_for(get_cute_dsl=_get_cute_dsl)


def restore_active_session_state_for(
    snapshot,
    *,
    get_cute_dsl: Callable[[], Any],
) -> None:
    if snapshot is None:
        pop_bundle_session(get_cute_dsl().compile_options)
        return

    compile_options, session_snapshot = snapshot
    if session_snapshot is None:
        pop_bundle_session(compile_options)
        return

    session = lookup_bundle_session(compile_options)
    if session is None:
        session = BundleSession()
        set_bundle_session(compile_options, session)
    session.restore(session_snapshot)


def restore_active_session_state(snapshot) -> None:
    restore_active_session_state_for(snapshot, get_cute_dsl=_get_cute_dsl)


def register_request(request: Any) -> None:
    active_bundle_session().add(request)


_DEFERRED_TEMP_STORAGE_VERSION = "3"


def _deferred_temp_storage_capability_error(cause: Exception | None = None) -> None:
    raise DSLRuntimeError(
        f"{_SESSION_SCOPE} deferred TempStorage requires "
        "a CUTLASS DSL with trace finalization, SmemAllocator, and MLIR value "
        "replacement support. Install a compatible CUTLASS DSL runtime "
        "separately; repository qualification uses the internal nightly.",
        cause=cause,
    )


def _active_cuda_kernel_op() -> Any:
    try:
        from cutlass._mlir import ir

        current_ip = ir.InsertionPoint.current
    except Exception as exc:
        raise DSLRuntimeError(
            f"{_SESSION_SCOPE} deferred TempStorage requires an active CuTe "
            "kernel trace."
        ) from exc

    if current_ip is None or current_ip.block is None:
        raise DSLRuntimeError(
            f"{_SESSION_SCOPE} deferred TempStorage requires an active CuTe "
            "kernel trace."
        )

    op = current_ip.block.owner
    while op is not None:
        operation = getattr(op, "operation", op)
        if getattr(operation, "name", None) == "cuda.kernel":
            return operation
        op = getattr(op, "parent_op", None) or getattr(op, "parent", None)

    raise DSLRuntimeError(
        f"{_SESSION_SCOPE} deferred TempStorage could not find the enclosing "
        "cuda.kernel operation."
    )


def _cuda_kernel_name(kernel_op: Any) -> str:
    try:
        return str(kernel_op.attributes["sym_name"])
    except Exception:
        return f"cuda.kernel@{id(kernel_op):x}"


def _fresh_i32_placeholder() -> Any:
    from cutlass._mlir.dialects import arith
    from cutlass.cutlass_dsl import T

    return arith.constant(T.i32(), 0)


def register_deferred_temp_storage_event(
    temp_storage: Any,
    *,
    primitive_name: str,
    requirement_key: Hashable,
    active_session_getter: Callable[[], BundleSession] = active_bundle_session,
) -> tuple[Any, Any, Any]:
    """Emit fresh ABI placeholders and record one deferred scratch use."""

    if not getattr(temp_storage, "is_deferred", False):
        raise ValueError("deferred scratch registration requires deferred TempStorage")
    try:
        hash(requirement_key)
    except TypeError as exc:
        raise TypeError("scratch requirement keys must be hashable") from exc

    session = active_session_getter()
    kernel_op = _active_cuda_kernel_op()
    smem_addr_placeholder = _fresh_i32_placeholder()
    size_placeholder = _fresh_i32_placeholder()
    try:
        location = str(smem_addr_placeholder.owner.location)
    except Exception:
        location = "unknown location"

    session.add_deferred_temp_storage_event(
        DeferredTempStorageEvent(
            kernel_op=kernel_op,
            kernel_name=_cuda_kernel_name(kernel_op),
            temp_storage=temp_storage,
            primitive_name=primitive_name,
            requirement_key=requirement_key,
            sharing=temp_storage.sharing,
            auto_sync=temp_storage.auto_sync,
            capacity_size_in_bytes=temp_storage.capacity_size_in_bytes,
            capacity_alignment=temp_storage.alignment,
            smem_addr_placeholder=smem_addr_placeholder,
            size_placeholder=size_placeholder,
            location=location,
        )
    )
    return (
        Uint32(smem_addr_placeholder),
        Int32(size_placeholder),
        Int32(1 if temp_storage.auto_sync else 0),
    )


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def plan_deferred_temp_storage_events(
    events: list[DeferredTempStorageEvent],
    layouts: Mapping[Hashable, ScratchLayout],
) -> tuple[DeferredTempStoragePlan, ...]:
    """Resolve exact kernel-local storage plans without mutating MLIR."""

    grouped: dict[tuple[Any, int], list[DeferredTempStorageEvent]] = {}
    for event in events:
        grouped.setdefault(
            (event.kernel_op, id(event.temp_storage)),
            [],
        ).append(event)

    plans: list[DeferredTempStoragePlan] = []
    for group_events in grouped.values():
        first = group_events[0]
        for event in group_events[1:]:
            if (
                event.sharing != first.sharing
                or event.auto_sync != first.auto_sync
                or event.capacity_size_in_bytes != first.capacity_size_in_bytes
                or event.capacity_alignment != first.capacity_alignment
            ):
                raise DSLRuntimeError(
                    "Deferred TempStorage configuration changed during tracing "
                    f"for {first.kernel_name} ({event.location})."
                )
        event_layouts = []
        for event in group_events:
            layout = layouts.get(event.requirement_key)
            if layout is None:
                raise DSLRuntimeError(
                    "No exact C++ scratch layout was registered for "
                    f"{event.primitive_name} ({event.location})."
                )
            event_layouts.append((event, layout))

        required_plan_alignment = max(layout.alignment for _, layout in event_layouts)
        if first.sharing == "shared":
            planned_size = max(layout.size_in_bytes for _, layout in event_layouts)
            planned_bindings = tuple(
                (event, 0, planned_size, required_plan_alignment)
                for event, _ in event_layouts
            )
        else:
            planned_size = 0
            exclusive_bindings = []
            for event, layout in event_layouts:
                offset = _align_up(planned_size, layout.alignment)
                exclusive_bindings.append(
                    (event, offset, layout.size_in_bytes, layout.alignment)
                )
                planned_size = offset + layout.size_in_bytes
            planned_bindings = tuple(exclusive_bindings)

        capacity_size = first.capacity_size_in_bytes
        if capacity_size is not None and capacity_size < planned_size:
            raise DSLRuntimeError(
                "Deferred TempStorage capacity is smaller than its resolved plan "
                f"in {first.kernel_name} ({capacity_size} < {planned_size})."
            )
        resolved_size_in_bytes = (
            capacity_size if capacity_size is not None else planned_size
        )
        if first.capacity_alignment is None:
            plan_alignment = required_plan_alignment
        else:
            if first.capacity_alignment < required_plan_alignment:
                raise DSLRuntimeError(
                    "Deferred TempStorage alignment is weaker than its resolved "
                    f"plan in {first.kernel_name}."
                )
            plan_alignment = first.capacity_alignment

        bindings = tuple(
            DeferredTempStorageBinding(
                event=event,
                byte_offset_in_bytes=offset,
                size_in_bytes=(
                    binding_size_in_bytes
                    if first.sharing == "exclusive"
                    else resolved_size_in_bytes
                ),
                alignment=(
                    binding_alignment
                    if first.sharing == "exclusive"
                    else plan_alignment
                ),
            )
            for event, offset, binding_size_in_bytes, binding_alignment in (
                planned_bindings
            )
        )
        plans.append(
            DeferredTempStoragePlan(
                kernel_op=first.kernel_op,
                kernel_name=first.kernel_name,
                temp_storage=first.temp_storage,
                size_in_bytes=resolved_size_in_bytes,
                alignment=plan_alignment,
                bindings=bindings,
            )
        )
    return tuple(plans)


def _replace_all_uses(old_value: Any, new_value: Any) -> None:
    for method_name in ("replace_all_uses_with", "replaceAllUsesWith"):
        replace = getattr(old_value, method_name, None)
        if replace is None:
            continue
        try:
            replace(new_value)
            return
        except Exception:
            continue
    _deferred_temp_storage_capability_error()


def materialize_deferred_temp_storage_plans(
    plans: tuple[DeferredTempStoragePlan, ...],
    module: Any,
) -> None:
    """Insert planned allocations and backpatch every recorded ABI operand."""

    if not plans:
        return

    try:
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import arith, llvm
        from cutlass.cute.typing import Pointer
        from cutlass.cutlass_dsl import T
        from cutlass.memory import SmemAllocator
    except (AttributeError, ImportError) as exc:
        _deferred_temp_storage_capability_error(exc)

    required_capabilities = (
        getattr(ir.InsertionPoint, "at_block_begin", None),
        getattr(SmemAllocator, "allocate", None),
        getattr(Pointer, "to_llvm_ptr", None),
        getattr(arith, "constant", None),
        getattr(arith, "addi", None),
        getattr(llvm, "ptrtoint", None),
    )
    if not all(callable(capability) for capability in required_capabilities):
        _deferred_temp_storage_capability_error()

    kernel_groups: dict[Any, tuple[Any, list[DeferredTempStoragePlan]]] = {}
    for plan in plans:
        try:
            entry_block = plan.kernel_op.regions[0].blocks[0]
        except Exception as exc:
            raise DSLRuntimeError(
                "Deferred TempStorage could not locate the entry block for "
                f"{plan.kernel_name}."
            ) from exc
        kernel_key = plan.kernel_op
        existing_group = kernel_groups.get(kernel_key)
        if existing_group is None:
            kernel_groups[kernel_key] = (entry_block, [plan])
        else:
            existing_group[1].append(plan)
        for binding in plan.bindings:
            for placeholder in (
                binding.event.smem_addr_placeholder,
                binding.event.size_placeholder,
            ):
                if not any(
                    callable(getattr(placeholder, method_name, None))
                    for method_name in ("replace_all_uses_with", "replaceAllUsesWith")
                ):
                    _deferred_temp_storage_capability_error()

    for entry_block, kernel_plans in kernel_groups.values():
        with ir.InsertionPoint.at_block_begin(entry_block):
            allocator = SmemAllocator()
            for plan in kernel_plans:
                smem_ptr = allocator.allocate(
                    plan.size_in_bytes,
                    plan.alignment,
                )
                base_addr = llvm.ptrtoint(T.i32(), smem_ptr.to_llvm_ptr())
                for binding in plan.bindings:
                    smem_addr = base_addr
                    if binding.byte_offset_in_bytes:
                        offset = arith.constant(
                            T.i32(),
                            binding.byte_offset_in_bytes,
                        )
                        smem_addr = arith.addi(base_addr, offset)
                    size = arith.constant(T.i32(), binding.size_in_bytes)
                    _replace_all_uses(
                        binding.event.smem_addr_placeholder,
                        smem_addr,
                    )
                    _replace_all_uses(binding.event.size_placeholder, size)

    module.operation.attributes["cuda_coop.cutlass.deferred_temp_storage_version"] = (
        ir.StringAttr.get(_DEFERRED_TEMP_STORAGE_VERSION)
    )


@dataclass(frozen=True)
class _TempStorageBinding:
    smem_addr_u32: Any
    size_in_bytes: int
    alignment: int
    auto_sync: bool


def materialize_temp_storage_binding(
    temp_storage: Any,
    *,
    scope: str = _SESSION_SCOPE,
    active_session_getter: Callable[[], BundleSession] = active_bundle_session,
    implicit_alignment: int = 8,
) -> _TempStorageBinding:
    session = active_session_getter()
    size_in_bytes = (
        temp_storage.capacity_size_in_bytes
        if temp_storage.capacity_size_in_bytes is not None
        else temp_storage.required_size_in_bytes
    )
    alignment = (
        temp_storage.alignment
        if temp_storage.alignment is not None
        else max(implicit_alignment, temp_storage.required_alignment)
    )

    binding = session.get_temp_storage_binding(temp_storage)
    # Primitive temp-storage requirements are discovered before provider
    # materialization. If a binding has already been allocated, growing it here
    # would leave earlier FFI call sites pointing at an undersized allocation.
    if (
        binding is not None
        and binding.size_in_bytes >= size_in_bytes
        and binding.alignment >= alignment
        and binding.auto_sync == temp_storage.auto_sync
    ):
        return binding
    if binding is not None:
        raise RuntimeError(
            f"{scope}.TempStorage requirements changed after shared-memory "
            "materialization; record all primitive uses before requesting "
            "provider storage"
        )

    binding = _TempStorageBinding(
        smem_addr_u32=_allocate_smem_addr_u32(size_in_bytes, alignment),
        size_in_bytes=size_in_bytes,
        alignment=alignment,
        auto_sync=temp_storage.auto_sync,
    )
    session.set_temp_storage_binding(temp_storage, binding)
    return binding


def _allocate_smem_addr_u32(size_in_bytes: int, alignment: int) -> Any:
    if size_in_bytes <= 0:
        return Uint32(0)

    from cutlass._mlir.dialects import llvm
    from cutlass.cute.arch import smem as cute_smem
    from cutlass.cutlass_dsl import T

    smem_ptr = cute_smem.alloc_smem(Uint8, size_in_bytes, alignment)
    # Carry shared-space address (u32) through ABI. Shims recover a usable
    # generic pointer via cvta.shared before doing typed accesses.
    return Uint32(llvm.ptrtoint(T.i32(), smem_ptr.to_llvm_ptr()))


def temp_storage_ffi_args_for_size(
    size_in_bytes: int,
    alignment: int,
    *,
    auto_sync: bool = True,
) -> tuple[Any, Any, Any]:
    return (
        _allocate_smem_addr_u32(size_in_bytes, alignment),
        Int32(size_in_bytes),
        Int32(1 if auto_sync else 0),
    )


def temp_storage_ffi_args(
    primitive_name: str,
    *,
    scope: str = _SESSION_SCOPE,
    active_session_getter: Callable[[], BundleSession] = active_bundle_session,
    implicit_alignment: int = 8,
) -> tuple[Any, Any, Any]:
    from ._single_phase import get_active_single_phase_context

    context = get_active_single_phase_context()
    temp_storage = context.temp_storage if context is not None else None
    if temp_storage is None:
        return (Uint32(0), Int32(0), Int32(1))
    if getattr(temp_storage, "is_deferred", False):
        raise NotImplementedError(
            "deferred TempStorage is currently supported only by "
            "cuda.coop.cutlass block Load, Store, Exchange, Scan, "
            "AdjacentDifference, Discontinuity, RadixSort, and MergeSort"
        )

    binding = materialize_temp_storage_binding(
        temp_storage,
        scope=scope,
        active_session_getter=active_session_getter,
        implicit_alignment=implicit_alignment,
    )
    primitive_slice = temp_storage.slice_for_latest_use(primitive_name)
    if primitive_slice is None or primitive_slice.size_in_bytes <= 0:
        return (Uint32(0), Int32(0), Int32(1 if binding.auto_sync else 0))

    smem_addr_arg = binding.smem_addr_u32
    slice_size_in_bytes = primitive_slice.size_in_bytes
    if temp_storage.sharing == "shared":
        slice_size_in_bytes = binding.size_in_bytes
    if primitive_slice.byte_offset_in_bytes != 0:
        smem_addr_arg = smem_addr_arg + Uint32(primitive_slice.byte_offset_in_bytes)
    return (
        smem_addr_arg,
        Int32(slice_size_in_bytes),
        Int32(1 if binding.auto_sync else 0),
    )


def clear_scalar_result_types() -> None:
    with _STATE_LOCK:
        _SCALAR_RESULT_TYPES.clear()


def canonical_dsl_type(
    value: Any,
    *,
    scope: str = _SESSION_SCOPE,
    root_scope: str = _SESSION_SCOPE,
) -> type:
    if isinstance(value, type) and value in TYPE_SPECS:
        return value

    if isinstance(value, np.dtype):
        ordinary_type = ORDINARY_PROVIDER_TYPES.get(value.type)
        if ordinary_type is not None:
            return ordinary_type

    if isinstance(value, type):
        ordinary_type = ORDINARY_PROVIDER_TYPES.get(value)
        if ordinary_type is not None:
            return ordinary_type

    value_type = type(value)
    ordinary_type = ORDINARY_PROVIDER_TYPES.get(value_type)
    if ordinary_type is not None:
        return ordinary_type
    if value_type in TYPE_SPECS:
        return value_type

    dsl_dtype = getattr(value, "dtype", None)
    if isinstance(dsl_dtype, type) and dsl_dtype in TYPE_SPECS:
        return dsl_dtype

    scalar_result_key = _scalar_result_type_key(value)
    with _STATE_LOCK:
        remembered_entry = _SCALAR_RESULT_TYPES.get(scalar_result_key)
    if remembered_entry is not None:
        remembered_value = (
            remembered_entry.strong_ref
            if remembered_entry.value_ref is None
            else remembered_entry.value_ref()
        )
        if remembered_value is value and remembered_entry.value_type in TYPE_SPECS:
            return remembered_entry.value_type

    mlir_type = getattr(value, "type", None)
    if mlir_type is None:
        return value_type

    ty_str = str(mlir_type)
    signed = getattr(value, "signed", None)
    is_float = bool(getattr(value, "is_float", False))

    if is_float:
        if ty_str == "f32":
            return Float32
        if ty_str == "f64":
            return Float64
        return value_type

    if ty_str in {"i32", "i64"} and signed is None:
        raise TypeError(
            f"{scope} provider cannot infer integer signedness; "
            f"pass a {root_scope}.ThreadData value, a CUDA DSL typing class, "
            "or a value with signed=True/False"
        )
    if ty_str == "i32":
        return Int32 if signed is True else Uint32
    if ty_str == "i64":
        return Int64 if signed is True else Uint64
    return value_type


def _validate_common_root_numeric_dtype(
    value: Any,
    *,
    operation: str | None = None,
) -> type:
    """Validate one CUTLASS value only while it crosses the common root."""

    value_type = canonical_dsl_type(value)
    if operation is None:
        operation = _common_root_operation_name()
    if operation is None:
        return value_type
    dtype_name = PROVIDER_TYPE_NAMES.get(value_type, "unsupported")
    validate_common_v1_numeric_dtype_name(dtype_name, operation=operation)
    return value_type


def _scalar_result_type_key(value: Any) -> tuple[int, type]:
    return id(value), type(value)


def _forget_scalar_result_type(
    key: tuple[int, type],
    value_ref: weakref.ReferenceType[Any],
) -> None:
    with _STATE_LOCK:
        entry = _SCALAR_RESULT_TYPES.get(key)
        if entry is not None and entry.value_ref is value_ref:
            _SCALAR_RESULT_TYPES.pop(key, None)


def remember_scalar_result_type(
    value: Any,
    value_type: type,
    *,
    scope: str = _SESSION_SCOPE,
    compile_options: Any = _COMPILE_OPTIONS_UNSET,
    compile_options_getter: Callable[[], Any] | None = None,
    group_metadata: ValueGroupMetadata | None = None,
) -> Any:
    if value_type not in TYPE_SPECS:
        return value
    if type(value) in _BUILTIN_SCALAR_VALUE_TYPES:
        if group_metadata is not None:
            raise RuntimeError(
                f"{scope} provider cannot attach group metadata to an interned "
                "builtin scalar result"
            )
        return value

    key = _scalar_result_type_key(value)
    with _STATE_LOCK:
        value_ref: weakref.ReferenceType[Any] | None
        strong_ref: Any | None = None
        finalizer = None
        try:
            value_ref = weakref.ref(value)
            finalizer = weakref.finalize(
                value,
                _forget_scalar_result_type,
                key,
                value_ref,
            )
        except TypeError:
            value_ref = None
            strong_ref = value
        else:
            if value_ref() is not value:
                if finalizer is not None:
                    finalizer.detach()
                return value

        session = None
        if strong_ref is not None:
            try:
                if compile_options is _COMPILE_OPTIONS_UNSET:
                    if compile_options_getter is None:
                        compile_options = _get_cute_dsl().compile_options
                    else:
                        compile_options = compile_options_getter()
                session = lookup_bundle_session(compile_options)
            except Exception:
                session = None
            if session is None:
                raise RuntimeError(
                    f"{scope} provider cannot remember a "
                    "non-weakrefable scalar result type without an active bundle session"
                )

        entry = _ScalarResultTypeEntry(
            value_type,
            value_ref,
            strong_ref,
            group_metadata,
        )
        _SCALAR_RESULT_TYPES[key] = entry
        if session is not None:
            session.add_scalar_result_type_entry(key, entry)
        if finalizer is not None:
            finalizer.atexit = False
    return value


def scalar_result_group_metadata(value: Any) -> ValueGroupMetadata | None:
    key = _scalar_result_type_key(value)
    with _STATE_LOCK:
        entry = _SCALAR_RESULT_TYPES.get(key)
    if entry is None:
        return None
    remembered_value = (
        entry.strong_ref if entry.value_ref is None else entry.value_ref()
    )
    if remembered_value is not value:
        return None
    return entry.group_metadata


register_scalar_metadata_lookup(scalar_result_group_metadata)


def validate_scan_reduce_op_for_type(
    op: str,
    value_type: type,
    *,
    root_scope: str,
    feature: str,
    namespace: str = "block",
) -> None:
    if op.startswith("bit_") and value_type not in RADIX_KEY_TYPES:
        operation = "reductions" if feature == "reduce" else "operations"
        raise TypeError(
            f"{root_scope}.{namespace}.{feature} bitwise {operation} require "
            "an integral type"
        )


def reduce_op_expr(op: str, lhs: str, rhs: str) -> str:
    if op == "sum":
        return f"{lhs} + {rhs}"
    if op == "multiplies":
        return f"{lhs} * {rhs}"
    if op == "min":
        return f"(({rhs}) < ({lhs}) ? ({rhs}) : ({lhs}))"
    if op == "max":
        return f"(({rhs}) > ({lhs}) ? ({rhs}) : ({lhs}))"
    if op == "bit_and":
        return f"{lhs} & {rhs}"
    if op == "bit_or":
        return f"{lhs} | {rhs}"
    if op == "bit_xor":
        return f"{lhs} ^ {rhs}"
    raise NotImplementedError(f"Unsupported reduce op: {op}")


def cub_op_expr(op: str) -> str:
    if op == "sum":
        return "::cuda::std::plus<>{}"
    if op == "multiplies":
        return "::cuda::std::multiplies<>{}"
    if op == "min":
        return "::cuda::minimum<>{}"
    if op == "max":
        return "::cuda::maximum<>{}"
    if op == "bit_and":
        return "::cuda::std::bit_and<>{}"
    if op == "bit_or":
        return "::cuda::std::bit_or<>{}"
    if op == "bit_xor":
        return "::cuda::std::bit_xor<>{}"
    raise NotImplementedError(f"Unsupported CUB op: {op}")


def bundle_source_preamble_lines(
    include_lines: list[str] | tuple[str, ...] = (),
) -> list[str]:
    return [
        *[line for line in include_lines if line],
        'extern "C" {',
        "static inline unsigned int cuda_coop_cutlass_lane_id() {",
        "  unsigned int lane;",
        '  asm("mov.u32 %0, %%laneid;" : "=r"(lane));',
        "  return lane;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_active_mask() {",
        "  unsigned int mask;",
        '  asm("activemask.b32 %0;" : "=r"(mask));',
        "  return mask;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_popc_u32(unsigned int value) {",
        "  unsigned int out;",
        '  asm("popc.b32 %0, %1;" : "=r"(out) : "r"(value));',
        "  return out;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_ballot_u32(",
        "    int predicate, unsigned int member_mask) {",
        "  unsigned int out;",
        (
            '  asm("{ .reg .pred p; setp.ne.u32 p, %1, 0; '
            'vote.sync.ballot.b32 %0, p, %2; }"'
            ' : "=r"(out) : "r"(predicate), "r"(member_mask));'
        ),
        "  return out;",
        "}",
        "static inline void cuda_coop_cutlass_warp_sync() {",
        "  unsigned int mask = cuda_coop_cutlass_active_mask();",
        '  asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");',
        "}",
        "static inline void cuda_coop_cutlass_block_sync() {",
        '  asm volatile("bar.sync 0;" ::: "memory");',
        "}",
        "static inline unsigned int cuda_coop_cutlass_tid_x() {",
        "  unsigned int tid;",
        '  asm("mov.u32 %0, %%tid.x;" : "=r"(tid));',
        "  return tid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_tid_y() {",
        "  unsigned int tid;",
        '  asm("mov.u32 %0, %%tid.y;" : "=r"(tid));',
        "  return tid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_tid_z() {",
        "  unsigned int tid;",
        '  asm("mov.u32 %0, %%tid.z;" : "=r"(tid));',
        "  return tid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_ntid_x() {",
        "  unsigned int ntid;",
        '  asm("mov.u32 %0, %%ntid.x;" : "=r"(ntid));',
        "  return ntid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_ntid_y() {",
        "  unsigned int ntid;",
        '  asm("mov.u32 %0, %%ntid.y;" : "=r"(ntid));',
        "  return ntid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_ntid_z() {",
        "  unsigned int ntid;",
        '  asm("mov.u32 %0, %%ntid.z;" : "=r"(ntid));',
        "  return ntid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_linear_tid() {",
        "  unsigned int tidx = cuda_coop_cutlass_tid_x();",
        "  unsigned int tidy = cuda_coop_cutlass_tid_y();",
        "  unsigned int tidz = cuda_coop_cutlass_tid_z();",
        "  unsigned int ntx = cuda_coop_cutlass_ntid_x();",
        "  unsigned int nty = cuda_coop_cutlass_ntid_y();",
        "  return tidx + ntx * (tidy + nty * tidz);",
        "}",
        "static inline unsigned int cuda_coop_cutlass_warps_in_block() {",
        "  unsigned int ntid = cuda_coop_cutlass_ntid_x() * cuda_coop_cutlass_ntid_y() *",
        "      cuda_coop_cutlass_ntid_z();",
        "  return (ntid + 31u) >> 5;",
        "}",
        "static inline int cuda_coop_cutlass_group_threads() {",
        (
            "  int group_threads = "
            "(int)(cuda_coop_cutlass_ntid_x() * cuda_coop_cutlass_ntid_y() * "
            "cuda_coop_cutlass_ntid_z());"
        ),
        "  if (group_threads <= 0) {",
        "    group_threads = 1;",
        "  }",
        "  return group_threads;",
        "}",
        "static inline void cuda_coop_cutlass_group_sync(int group_threads) {",
        "  if (group_threads > 32) {",
        "    cuda_coop_cutlass_block_sync();",
        "    return;",
        "  }",
        "  cuda_coop_cutlass_warp_sync();",
        "}",
        "static inline void* cuda_coop_cutlass_shared_ptr(",
        "    unsigned int shared_addr) {",
        "  unsigned long long shared_addr_u64 =",
        "      static_cast<unsigned long long>(shared_addr);",
        "  unsigned long long generic_addr;",
        '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_addr) : "l"(shared_addr_u64));',
        "  return reinterpret_cast<void*>(generic_addr);",
        "}",
        "static inline int cuda_coop_cutlass_sort_take_peer(",
        "    int peer_before, int equal, int low_slot) {",
        "  return low_slot ? peer_before : (!peer_before && !equal);",
        "}",
        "static inline int cuda_coop_cutlass_sort_rank_increment(",
        "    int before, int equal, int local_lane, int peer_lane) {",
        "  return before || (equal && local_lane > peer_lane);",
        "}",
        "static inline int cuda_coop_cutlass_use_temp_storage(",
        "    unsigned int temp_storage_smem_addr, int temp_storage_bytes,",
        "    int temp_storage_auto_sync, unsigned int bytes_per_lane) {",
        "  (void)temp_storage_smem_addr;",
        "  (void)temp_storage_auto_sync;",
        "  unsigned long long required_temp_bytes =",
        "      (unsigned long long)bytes_per_lane * 32ull *",
        "      (unsigned long long)cuda_coop_cutlass_warps_in_block();",
        "  int has_required_bytes = temp_storage_bytes > 0 &&",
        "      (unsigned long long)temp_storage_bytes >= required_temp_bytes;",
        "  if (bytes_per_lane > 0u && cuda_coop_cutlass_group_threads() > 32 &&",
        "      !has_required_bytes) {",
        '    asm volatile("trap;");',
        "  }",
        "  return has_required_bytes;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_pair_bytes_per_lane(",
        "    unsigned int first_bytes_per_lane, unsigned int second_bytes_per_lane,",
        "    unsigned int second_alignment) {",
        "  unsigned int storage_threads =",
        "      32u * cuda_coop_cutlass_warps_in_block();",
        "  unsigned int first_region_size = storage_threads * first_bytes_per_lane;",
        "  unsigned int second_offset =",
        "      (first_region_size + second_alignment - 1u) &",
        "      ~(second_alignment - 1u);",
        "  unsigned int total_bytes =",
        "      second_offset + storage_threads * second_bytes_per_lane;",
        "  return (total_bytes + storage_threads - 1u) / storage_threads;",
        "}",
        "static inline void cuda_coop_cutlass_st_shared_u32(",
        "    unsigned int base_addr, unsigned int idx, unsigned int value) {",
        "  unsigned int addr = base_addr + (idx << 2);",
        (
            '  asm volatile("st.shared.b32 [%0], %1;" '
            ':: "r"(addr), "r"(value) : "memory");'
        ),
        "}",
        "static inline unsigned int cuda_coop_cutlass_ld_shared_u32(",
        "    unsigned int base_addr, unsigned int idx) {",
        "  unsigned int addr = base_addr + (idx << 2);",
        "  unsigned int out;",
        '  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(out) : "r"(addr));',
        "  return out;",
        "}",
        "static inline void cuda_coop_cutlass_st_shared_u8(",
        "    unsigned int base_addr, unsigned int idx, unsigned char value) {",
        "  cuda_coop_cutlass_st_shared_u32(base_addr, idx, (unsigned int)value);",
        "}",
        "static inline unsigned char cuda_coop_cutlass_ld_shared_u8(",
        "    unsigned int base_addr, unsigned int idx) {",
        "  return (unsigned char)cuda_coop_cutlass_ld_shared_u32(base_addr, idx);",
        "}",
        "static inline void cuda_coop_cutlass_st_shared_u64(",
        ("    unsigned int base_addr, unsigned int idx, unsigned long long value) {"),
        "  unsigned int addr = base_addr + (idx << 3);",
        (
            '  asm volatile("st.shared.b64 [%0], %1;" '
            ':: "r"(addr), "l"(value) : "memory");'
        ),
        "}",
        "static inline unsigned long long cuda_coop_cutlass_ld_shared_u64(",
        "    unsigned int base_addr, unsigned int idx) {",
        "  unsigned int addr = base_addr + (idx << 3);",
        "  unsigned long long out;",
        '  asm volatile("ld.shared.b64 %0, [%1];" : "=l"(out) : "r"(addr));',
        "  return out;",
        "}",
        "static inline void cuda_coop_cutlass_st_shared_i32(",
        "    unsigned int base_addr, unsigned int idx, int value) {",
        ("  cuda_coop_cutlass_st_shared_u32(base_addr, idx, (unsigned int)value);"),
        "}",
        "static inline int cuda_coop_cutlass_ld_shared_i32(",
        "    unsigned int base_addr, unsigned int idx) {",
        ("  return (int)cuda_coop_cutlass_ld_shared_u32(base_addr, idx);"),
        "}",
        "static inline void cuda_coop_cutlass_st_shared_i64(",
        "    unsigned int base_addr, unsigned int idx, long long value) {",
        (
            "  cuda_coop_cutlass_st_shared_u64("
            "base_addr, idx, (unsigned long long)value);"
        ),
        "}",
        "static inline long long cuda_coop_cutlass_ld_shared_i64(",
        "    unsigned int base_addr, unsigned int idx) {",
        ("  return (long long)cuda_coop_cutlass_ld_shared_u64(base_addr, idx);"),
        "}",
        "static inline void cuda_coop_cutlass_st_shared_f32(",
        "    unsigned int base_addr, unsigned int idx, float value) {",
        "  union { float f; unsigned int u; } bits;",
        "  bits.f = value;",
        "  cuda_coop_cutlass_st_shared_u32(base_addr, idx, bits.u);",
        "}",
        "static inline float cuda_coop_cutlass_ld_shared_f32(",
        "    unsigned int base_addr, unsigned int idx) {",
        "  union { float f; unsigned int u; } bits;",
        "  bits.u = cuda_coop_cutlass_ld_shared_u32(base_addr, idx);",
        "  return bits.f;",
        "}",
        "static inline void cuda_coop_cutlass_st_shared_f64(",
        "    unsigned int base_addr, unsigned int idx, double value) {",
        "  union { double d; unsigned long long u; } bits;",
        "  bits.d = value;",
        "  cuda_coop_cutlass_st_shared_u64(base_addr, idx, bits.u);",
        "}",
        "static inline double cuda_coop_cutlass_ld_shared_f64(",
        "    unsigned int base_addr, unsigned int idx) {",
        "  union { double d; unsigned long long u; } bits;",
        "  bits.u = cuda_coop_cutlass_ld_shared_u64(base_addr, idx);",
        "  return bits.d;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_shfl_up_u32(",
        "    unsigned int value, int offset) {",
        "  unsigned int out;",
        "  unsigned int mask = cuda_coop_cutlass_active_mask();",
        (
            '  asm("shfl.sync.up.b32 %0, %1, %2, 0, %3;"'
            ' : "=r"(out) : "r"(value), "r"(offset), "r"(mask));'
        ),
        "  return out;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_shfl_idx_u32(",
        "    unsigned int value, int src_lane) {",
        "  unsigned int out;",
        "  unsigned int mask = cuda_coop_cutlass_active_mask();",
        (
            '  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;"'
            ' : "=r"(out) : "r"(value), "r"(src_lane), "r"(mask));'
        ),
        "  return out;",
        "}",
        (
            "static inline unsigned long long cuda_coop_cutlass_shfl_up_u64("
            "unsigned long long value, int offset) {"
        ),
        "  unsigned int lo = (unsigned int)value;",
        "  unsigned int hi = (unsigned int)(value >> 32);",
        "  lo = cuda_coop_cutlass_shfl_up_u32(lo, offset);",
        "  hi = cuda_coop_cutlass_shfl_up_u32(hi, offset);",
        "  return ((unsigned long long)hi << 32) | (unsigned long long)lo;",
        "}",
        (
            "static inline unsigned long long cuda_coop_cutlass_shfl_idx_u64("
            "unsigned long long value, int src_lane) {"
        ),
        "  unsigned int lo = (unsigned int)value;",
        "  unsigned int hi = (unsigned int)(value >> 32);",
        "  lo = cuda_coop_cutlass_shfl_idx_u32(lo, src_lane);",
        "  hi = cuda_coop_cutlass_shfl_idx_u32(hi, src_lane);",
        "  return ((unsigned long long)hi << 32) | (unsigned long long)lo;",
        "}",
        "static inline int cuda_coop_cutlass_shfl_up_i32(int value, int offset) {",
        "  return (int)cuda_coop_cutlass_shfl_up_u32((unsigned int)value, offset);",
        "}",
        "static inline int cuda_coop_cutlass_shfl_idx_i32(int value, int src_lane) {",
        "  return (int)cuda_coop_cutlass_shfl_idx_u32((unsigned int)value, src_lane);",
        "}",
        "static inline unsigned char cuda_coop_cutlass_shfl_up_u8(",
        "    unsigned char value, int offset) {",
        (
            "  return (unsigned char)cuda_coop_cutlass_shfl_up_u32("
            "(unsigned int)value, offset);"
        ),
        "}",
        "static inline unsigned char cuda_coop_cutlass_shfl_idx_u8(",
        "    unsigned char value, int src_lane) {",
        (
            "  return (unsigned char)cuda_coop_cutlass_shfl_idx_u32("
            "(unsigned int)value, src_lane);"
        ),
        "}",
        (
            "static inline long long cuda_coop_cutlass_shfl_up_i64("
            "long long value, int offset) {"
        ),
        (
            "  return (long long)cuda_coop_cutlass_shfl_up_u64("
            "(unsigned long long)value, offset);"
        ),
        "}",
        (
            "static inline long long cuda_coop_cutlass_shfl_idx_i64("
            "long long value, int src_lane) {"
        ),
        (
            "  return (long long)cuda_coop_cutlass_shfl_idx_u64("
            "(unsigned long long)value, src_lane);"
        ),
        "}",
        "static inline float cuda_coop_cutlass_shfl_up_f32(float value, int offset) {",
        "  union { float f; unsigned int u; } in_v, out_v;",
        "  in_v.f = value;",
        "  out_v.u = cuda_coop_cutlass_shfl_up_u32(in_v.u, offset);",
        "  return out_v.f;",
        "}",
        (
            "static inline float cuda_coop_cutlass_shfl_idx_f32("
            "float value, int src_lane) {"
        ),
        "  union { float f; unsigned int u; } in_v, out_v;",
        "  in_v.f = value;",
        "  out_v.u = cuda_coop_cutlass_shfl_idx_u32(in_v.u, src_lane);",
        "  return out_v.f;",
        "}",
        "static inline double cuda_coop_cutlass_shfl_up_f64(double value, int offset) {",
        "  union { double d; unsigned long long u; } in_v, out_v;",
        "  in_v.d = value;",
        "  out_v.u = cuda_coop_cutlass_shfl_up_u64(in_v.u, offset);",
        "  return out_v.d;",
        "}",
        (
            "static inline double cuda_coop_cutlass_shfl_idx_f64("
            "double value, int src_lane) {"
        ),
        "  union { double d; unsigned long long u; } in_v, out_v;",
        "  in_v.d = value;",
        "  out_v.u = cuda_coop_cutlass_shfl_idx_u64(in_v.u, src_lane);",
        "  return out_v.d;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_extract_sort_bits_u32(",
        "    unsigned int key, int begin_bit, int end_bit) {",
        "  int span = end_bit - begin_bit;",
        "  unsigned int mask = (span >= 32) ? 0xffffffffu : ((1u << span) - 1u);",
        "  return (key >> begin_bit) & mask;",
        "}",
        (
            "static inline unsigned long long cuda_coop_cutlass_extract_sort_bits_u64("
            "unsigned long long key, int begin_bit, int end_bit) {"
        ),
        "  int span = end_bit - begin_bit;",
        (
            "  unsigned long long mask = (span >= 64) ? 0xffffffffffffffffull "
            ": ((1ull << span) - 1ull);"
        ),
        "  return (key >> begin_bit) & mask;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_radix_order_u32_from_i32(",
        "    int key) {",
        "  return ((unsigned int)key) ^ 0x80000000u;",
        "}",
        (
            "static inline unsigned long long cuda_coop_cutlass_radix_order_u64_from_i64("
            "long long key) {"
        ),
        "  return ((unsigned long long)key) ^ 0x8000000000000000ull;",
        "}",
    ]


def render_bundle_source(
    requests: list[Any],
    *,
    scope: str,
    include_lines: list[str] | tuple[str, ...] = (),
    multi_item_kinds: frozenset[str] = frozenset(),
    render_local_request: Callable[[Any], list[str]],
) -> str:
    """Render a CuTe provider bundle with provider-specific local requests.

    Providers pass their extra include lines, the request kinds that support
    multi-item ``ThreadData``, and a ``render_local_request`` callback for
    requests that are not handled by registered shared renderers. The callback
    must return C++ source lines for supported local request kinds and raise for
    unsupported kinds.
    """
    requests = list(canonical_bundle_requests(requests))
    preamble_lines = canonical_bundle_preamble_lines(
        [
            *include_lines,
            *bundle_include_lines(requests),
        ]
    )
    lines = bundle_source_preamble_lines(preamble_lines)

    for request in requests:
        if bundle_renderer_for(request) is not None:
            continue
        if request.items_per_thread != 1 and request.kind not in multi_item_kinds:
            raise NotImplementedError(
                f"{scope} provider does not yet support multi-item "
                "ThreadData shims; use items_per_thread=1"
            )

    for request in requests:
        external_renderer = bundle_renderer_for(request)
        if external_renderer is not None:
            lines.extend(external_renderer.render(request))
            continue
        lines.extend(render_local_request(request))

    lines.append("}")
    return "\n".join(lines) + "\n"


def resolve_provider_type(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
    root_scope: str,
    namespace: str,
    canonical_type: Callable[[Any], type],
) -> type:
    value_type = canonical_type(value)
    operation = _common_root_operation_name()
    if operation is not None:
        dtype_name = PROVIDER_TYPE_NAMES.get(value_type, "unsupported")
        validate_common_v1_numeric_dtype_name(dtype_name, operation=operation)
    if value_type not in TYPE_SPECS or value_type not in allowed:
        raise NotImplementedError(
            f"{root_scope}.{namespace} provider {feature} currently supports "
            f"{supported_names(allowed)} only"
        )
    return value_type


def make_provider_type_resolver(
    *,
    scope: str,
    root_scope: str,
    namespace: str,
) -> Callable[..., type]:
    def resolve_type(
        value: Any,
        *,
        allowed: frozenset[type],
        feature: str,
    ) -> type:
        return resolve_provider_type(
            value,
            allowed=allowed,
            feature=feature,
            root_scope=root_scope,
            namespace=namespace,
            canonical_type=lambda value: canonical_dsl_type(
                value,
                scope=scope,
                root_scope=root_scope,
            ),
        )

    return resolve_type


def make_scalar_type_resolver(
    *,
    scope: str,
    resolve_type: Callable[..., type],
    allowed: frozenset[type] = SCAN_REDUCE_TYPES,
) -> Callable[..., type]:
    def resolve_scalar_type(value: Any, *, feature: str) -> type:
        if isinstance(value, ThreadData):
            raise TypeError(f"{scope}.{feature} currently expects a scalar value")
        return resolve_type(
            value,
            allowed=allowed,
            feature=feature,
        )

    return resolve_scalar_type


def as_int32_bool(value: bool) -> Any:
    return Int32(1 if value else 0)


def as_int32(value: Any) -> Any:
    if isinstance(value, Int32):
        return value
    return Int32(value)


def as_valid_items_arg(value: Any, *, scope: str) -> Any:
    if value is None:
        return Int32(-1)
    if isinstance(value, Int32):
        return value
    try:
        return Int32(value)
    except Exception as exc:
        raise TypeError(f"{scope} valid_items must be convertible to Int32") from exc


def _static_int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        raise TypeError("radix bit bounds must be int-like scalars")
    if isinstance(value, Integral):
        return int(value)
    return None


def validate_radix_bit_range(
    begin_bit: Any,
    end_bit: Any | None,
    key_type: type,
) -> Any:
    width_bits = TYPE_SPECS[key_type].width_bits
    resolved_end_bit = width_bits if end_bit is None else end_bit

    static_begin_bit = _static_int_value(begin_bit)
    static_end_bit = _static_int_value(resolved_end_bit)

    if static_begin_bit is not None and static_begin_bit < 0:
        raise ValueError("begin_bit must be non-negative")
    if static_begin_bit is not None and static_begin_bit >= width_bits:
        raise ValueError(f"begin_bit must be < {width_bits}")
    if (
        static_begin_bit is not None
        and static_end_bit is not None
        and static_end_bit <= static_begin_bit
    ):
        raise ValueError("end_bit must be greater than begin_bit")
    if static_end_bit is not None and static_end_bit > width_bits:
        raise ValueError(f"end_bit must be <= {width_bits}")
    return resolved_end_bit


def type_size_bytes(value_type: type) -> int:
    return max(1, (TYPE_SPECS[value_type].width_bits + 7) // 8)


def coerce_scan_initial_value(
    *,
    initial_value: Any,
    value_type: type,
    root_scope: str,
    feature: str,
    namespace: str,
) -> Any:
    if isinstance(initial_value, value_type):
        return initial_value
    try:
        return value_type(initial_value)
    except Exception as exc:
        raise TypeError(
            f"{root_scope}.{namespace}.{feature} initial_value cannot be converted to "
            f"{value_type.__name__}"
        ) from exc


def resolve_thread_data_value_type(
    value: ThreadData,
    *,
    allowed: frozenset[type],
    feature: str,
    scope: str,
    resolve_type: Callable[..., type],
    supported_types: frozenset[type] = ALL_PROVIDER_TYPES,
) -> tuple[type, tuple[Any, ...]]:
    values = value.values(feature)
    if value.dtype is not None:
        value_type = resolve_type(value.dtype, allowed=allowed, feature=feature)
        converted: list[Any] = []
        for idx, item in enumerate(values):
            plain_item = coerce_plain_scalar(
                item,
                value_type,
                name=f"{feature} ThreadData item {idx}",
                scope=scope,
                allow_nonfinite=True,
            )
            if plain_item is not _NOT_PLAIN_SCALAR:
                converted.append(plain_item)
                continue
            try:
                item_type = resolve_type(
                    item,
                    allowed=supported_types,
                    feature=feature,
                )
            except TypeError as exc:
                if _signless_integer_item_matches_dtype(item, value_type):
                    converted.append(item)
                    continue
                raise TypeError(
                    f"{scope}.{feature} ThreadData item {idx} type "
                    "cannot be reconciled with declared dtype"
                ) from exc
            except NotImplementedError as exc:
                raise TypeError(
                    f"{scope}.{feature} ThreadData item {idx} type "
                    "cannot be reconciled with declared dtype"
                ) from exc
            if item_type is not value_type:
                raise TypeError(
                    f"{scope}.{feature} ThreadData dtype does not match "
                    "initialized item types"
                )
            converted.append(item)
        return value_type, tuple(converted)

    value_type = resolve_type(values[0], allowed=allowed, feature=feature)
    for item in values[1:]:
        item_type = resolve_type(item, allowed=allowed, feature=feature)
        if item_type is not value_type:
            raise TypeError(
                f"{scope}.{feature} ThreadData requires homogeneous item types"
            )
    return value_type, values


def _signless_integer_item_matches_dtype(item: Any, value_type: type) -> bool:
    if value_type not in {Int32, Uint32, Int64, Uint64}:
        return False
    if getattr(item, "signed", None) is not None:
        return False
    mlir_type = getattr(item, "type", None)
    if mlir_type is None:
        return False
    return str(mlir_type) == f"i{TYPE_SPECS[value_type].width_bits}"


def make_thread_data_value_type_resolver(
    *,
    scope: str,
    resolve_type: Callable[..., type],
    supported_types: frozenset[type] = ALL_PROVIDER_TYPES,
) -> Callable[..., tuple[type, tuple[Any, ...]]]:
    def resolve_value_type(
        value: ThreadData,
        *,
        allowed: frozenset[type],
        feature: str,
    ) -> tuple[type, tuple[Any, ...]]:
        return resolve_thread_data_value_type(
            value,
            allowed=allowed,
            feature=feature,
            scope=scope,
            resolve_type=resolve_type,
            supported_types=supported_types,
        )

    return resolve_value_type


def resolve_thread_data_pair_types(
    *,
    key: Any,
    value: Any,
    allowed_key_types: frozenset[type],
    allowed_value_types: frozenset[type],
    feature: str,
    scope: str,
    resolve_type: Callable[..., type],
) -> tuple[type, tuple[Any, ...], ThreadData, type, tuple[Any, ...], ThreadData]:
    if isinstance(key, ThreadData) or isinstance(value, ThreadData):
        if not isinstance(key, ThreadData) or not isinstance(value, ThreadData):
            raise TypeError(
                f"{scope}.{feature} requires both key and value to be "
                "ThreadData when one argument uses ThreadData"
            )
        if key.items_per_thread != value.items_per_thread:
            raise ValueError(
                f"{scope}.{feature} requires matching "
                "ThreadData.items_per_thread for key and value"
            )

    if not isinstance(key, ThreadData) or not isinstance(value, ThreadData):
        raise TypeError(
            f"{scope}.{feature} internal ThreadData path requires "
            "ThreadData key/value inputs"
        )

    key_type, key_values = resolve_thread_data_value_type(
        key,
        allowed=allowed_key_types,
        feature=feature,
        scope=scope,
        resolve_type=resolve_type,
    )
    value_type, value_values = resolve_thread_data_value_type(
        value,
        allowed=allowed_value_types,
        feature=feature,
        scope=scope,
        resolve_type=resolve_type,
    )
    return key_type, key_values, key, value_type, value_values, value


def require_single_item_thread_data(
    feature: str,
    *thread_data_args: ThreadData,
    scope: str,
) -> None:
    for thread_data in thread_data_args:
        if thread_data.items_per_thread != 1:
            raise NotImplementedError(
                f"{scope}.{feature} does not yet support multi-item "
                "ThreadData provider shims; use items_per_thread=1"
            )


def validate_thread_data_output(
    *,
    output: Any,
    expected_items_per_thread: int,
    resolved_dtype: type,
    scope: str,
    primitive_name: str,
    output_name: str,
    resolve_type: Callable[..., type],
    assigned_dtype: Any | None = None,
    type_label: str = "ThreadData",
    item_count_message: str | None = None,
) -> ThreadData | None:
    if output is None:
        return None
    if not isinstance(output, ThreadData):
        raise TypeError(f"{scope}.{primitive_name} {output_name} must be {type_label}")
    if output.items_per_thread != expected_items_per_thread:
        if item_count_message is None:
            item_count_message = (
                f"{scope}.{primitive_name} {output_name} must have "
                f"items_per_thread={expected_items_per_thread}"
            )
        raise ValueError(item_count_message)
    if output.dtype is not None:
        resolve_type(
            output.dtype,
            allowed=frozenset({resolved_dtype}),
            feature=primitive_name,
        )
    else:
        output.dtype = resolved_dtype if assigned_dtype is None else assigned_dtype
    return output


def thread_data_output_dtype(value: ThreadData, value_type: type) -> Any:
    return value.dtype if value.dtype is not None else value_type

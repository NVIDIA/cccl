# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Per-trace provider sessions, finalizer activation, and scalar identity."""

from __future__ import annotations

import importlib
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.typing import (
    Float32,
    Float64,
    Int32,
    Int64,
    Uint32,
    Uint64,
)

from cuda.coop._core.api import _common_root_operation_name
from cuda.coop._core.dtype_policy import validate_portable_numeric_dtype_name

from .._value_metadata import (
    ValueGroupMetadata,
    register_scalar_metadata_lookup,
)
from ._rendering import canonical_bundle_requests
from ._types import (
    ORDINARY_PROVIDER_TYPES,
    PROVIDER_TYPE_NAMES,
    TYPE_SPECS,
    DeferredTempStorageEvent,
    supported_names,
)


@dataclass(frozen=True)
class _ScalarResultTypeEntry:
    value_type: type
    value_ref: weakref.ReferenceType[Any] | None
    strong_ref: Any | None = None
    group_metadata: ValueGroupMetadata | None = None


_SESSION_SCOPE = "cuda.coop.cutlass"

_TRACE_HOOK_DISPATCHER_ATTR = "_cuda_coop_cutlass_provider_trace_finalize_dispatcher"

_TRACE_HOOK_TARGET_ATTR = "_cuda_coop_cutlass_provider_trace_finalize_hook"

_BUILTIN_SCALAR_VALUE_TYPES = (bool, int, float, complex, str)

_BUNDLE_FINALIZER: Callable[[Any, Any, str], None] | None = None

_COMPILE_OPTIONS_UNSET = object()

_STATE_LOCK = threading.RLock()

_SCALAR_RESULT_TYPES: dict[tuple[int, type], _ScalarResultTypeEntry] = {}

_SESSIONS: weakref.WeakKeyDictionary[Any, BundleSession] = weakref.WeakKeyDictionary()

_ID_SESSIONS: dict[int, tuple[weakref.ReferenceType[Any], BundleSession, Any]] = {}


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
        with _STATE_LOCK, self._lock:
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
        importlib.import_module(f"{__package__}._finalize")
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
                "runtime with register_trace_finalize_hook and link-libraries "
                "support."
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
    validate_portable_numeric_dtype_name(dtype_name, operation=operation)
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
        validate_portable_numeric_dtype_name(dtype_name, operation=operation)
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

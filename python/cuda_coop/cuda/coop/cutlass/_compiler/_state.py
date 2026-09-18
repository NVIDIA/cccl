# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Per-trace provider sessions and finalizer ownership."""

from __future__ import annotations

import importlib
import threading
import weakref
from collections.abc import Callable
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

from ._rendering import canonical_bundle_requests

_SESSION_SCOPE = "cuda.coop.cutlass"
_TRACE_HOOK_DISPATCHER_ATTR = "_cuda_coop_cutlass_provider_trace_finalize_dispatcher"
_TRACE_HOOK_TARGET_ATTR = "_cuda_coop_cutlass_provider_trace_finalize_hook"
_BUNDLE_FINALIZER: Callable[[Any, Any, str], None] | None = None
_STATE_LOCK = threading.RLock()
_SESSIONS: weakref.WeakKeyDictionary[Any, list[BundleSession]] = (
    weakref.WeakKeyDictionary()
)
_ID_SESSIONS: dict[
    int, tuple[weakref.ReferenceType[Any], list[BundleSession], weakref.finalize]
] = {}
_UNSPECIFIED_MODULE = object()


class BundleSession:
    def __init__(self, trace_module_op=None):
        self.trace_module_op = trace_module_op
        self.requests = set()
        self._lock = threading.RLock()

    def add(self, request):
        with self._lock:
            self.requests.add(request)

    def snapshot(self):
        with self._lock:
            return self.trace_module_op, set(self.requests)

    def restore(self, snapshot):
        with self._lock:
            self.trace_module_op, requests = snapshot
            self.requests = set(requests)

    def request_list(self):
        with self._lock:
            return list(canonical_bundle_requests(self.requests))

    def is_empty(self):
        with self._lock:
            return not self.requests

    def belongs_to_trace_module(self, module):
        with self._lock:
            return _same_mlir_operation(self.trace_module_op, module)

    def bind_trace_module(self, module):
        with self._lock:
            if self.trace_module_op is None:
                self.trace_module_op = module
            return self.belongs_to_trace_module(module)


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
    with _STATE_LOCK:
        if getattr(dsl, _TRACE_HOOK_DISPATCHER_ATTR, None) is None:
            register_hook = getattr(dsl, "register_trace_finalize_hook", None)
            if register_hook is None:
                raise DSLRuntimeError(
                    f"{scope} provider requires CuTe DSL trace-finalize hook "
                    "support so generated cooperative-primitive bundles can be "
                    "linked into the compiled kernel. Install a compatible "
                    "CUTLASS DSL runtime with register_trace_finalize_hook and "
                    "link-libraries support."
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


def _sessions_for_options(compile_options: Any) -> list[BundleSession] | None:
    try:
        return _SESSIONS.get(compile_options)
    except TypeError:
        entry = _ID_SESSIONS.get(id(compile_options))
        if entry is None:
            return None
        options_ref, sessions, finalizer = entry
        if options_ref() is compile_options:
            return sessions
        if finalizer.alive:
            finalizer.detach()
        _ID_SESSIONS.pop(id(compile_options), None)
        return None


def _select_session(
    sessions: list[BundleSession], trace_module_op: Any
) -> BundleSession | None:
    if trace_module_op is _UNSPECIFIED_MODULE:
        return sessions[0] if len(sessions) == 1 else None
    return next(
        (
            session
            for session in sessions
            if session.belongs_to_trace_module(trace_module_op)
        ),
        None,
    )


def lookup_bundle_session(
    compile_options: Any, *, trace_module_op: Any = _UNSPECIFIED_MODULE
) -> BundleSession | None:
    """Find a trace's session without disturbing other modules on the same DSL."""

    with _STATE_LOCK:
        return _select_session(
            _sessions_for_options(compile_options) or [], trace_module_op
        )


def _drop_id_session(key: int) -> None:
    with _STATE_LOCK:
        _ID_SESSIONS.pop(key, None)


def _store_bundle_sessions(compile_options: Any, sessions: list[BundleSession]) -> None:
    try:
        _SESSIONS[compile_options] = sessions
    except TypeError:
        try:
            options_ref = weakref.ref(compile_options)
        except TypeError as exc:
            raise DSLRuntimeError(
                f"{_SESSION_SCOPE} provider compile_options must be weak-referenceable."
            ) from exc
        key = id(compile_options)
        finalizer = weakref.finalize(compile_options, _drop_id_session, key)
        _ID_SESSIONS[key] = (options_ref, sessions, finalizer)


def set_bundle_session(compile_options: Any, session: BundleSession) -> None:
    with _STATE_LOCK:
        sessions = _sessions_for_options(compile_options)
        if sessions is None:
            _store_bundle_sessions(compile_options, [session])
            return
        previous = _select_session(sessions, session.trace_module_op)
        if previous is not None:
            sessions.remove(previous)
        sessions.append(session)


def pop_bundle_session(
    compile_options: Any, *, trace_module_op: Any = _UNSPECIFIED_MODULE
) -> BundleSession | None:
    """Remove only the matching trace, leaving nested or outer traces intact."""

    with _STATE_LOCK:
        sessions = _sessions_for_options(compile_options)
        if not sessions:
            return None
        session = _select_session(sessions, trace_module_op)
        if session is None:
            return None
        sessions.remove(session)
        if not sessions:
            try:
                _SESSIONS.pop(compile_options, None)
            except TypeError:
                entry = _ID_SESSIONS.pop(id(compile_options), None)
                if entry is not None and entry[2].alive:
                    entry[2].detach()
        return session


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
        session = lookup_bundle_session(
            compile_options, trace_module_op=trace_module_op
        )
        if session is None and trace_module_op is not None:
            unbound = lookup_bundle_session(compile_options, trace_module_op=None)
            if unbound is not None:
                unbound.bind_trace_module(trace_module_op)
                session = unbound
        if session is None:
            session = BundleSession(trace_module_op=trace_module_op)
            set_bundle_session(compile_options, session)
        return session


def active_bundle_session() -> BundleSession:
    module = _active_trace_module_op()
    if module is None:
        raise DSLRuntimeError(
            f"{_SESSION_SCOPE} provider requires an active CuTe trace module."
        )
    _ensure_trace_hook_registered()
    compile_options = _get_cute_dsl().compile_options
    return get_or_create_bundle_session(
        compile_options,
        trace_module_op=module,
    )


def snapshot_active_session_state_for(*, get_cute_dsl: Callable[[], Any]):
    compile_options = get_cute_dsl().compile_options
    module = _active_trace_module_op()
    with _STATE_LOCK:
        session = lookup_bundle_session(compile_options, trace_module_op=module)
        return compile_options, module, None if session is None else session.snapshot()


def snapshot_active_session_state():
    return snapshot_active_session_state_for(get_cute_dsl=_get_cute_dsl)


def restore_active_session_state_for(
    snapshot,
    *,
    get_cute_dsl: Callable[[], Any],
) -> None:
    current_options = get_cute_dsl().compile_options
    current_module = _active_trace_module_op()
    with _STATE_LOCK:
        if snapshot is None:
            pop_bundle_session(current_options, trace_module_op=current_module)
            return
        compile_options, module, session_snapshot = snapshot
        if current_options is not compile_options:
            pop_bundle_session(current_options, trace_module_op=current_module)
        if session_snapshot is None:
            pop_bundle_session(compile_options, trace_module_op=module)
            return
        session = get_or_create_bundle_session(compile_options, trace_module_op=module)
        session.restore(session_snapshot)


def restore_active_session_state(snapshot) -> None:
    restore_active_session_state_for(snapshot, get_cute_dsl=_get_cute_dsl)


def register_request(request: Any) -> None:
    active_bundle_session().add(request)

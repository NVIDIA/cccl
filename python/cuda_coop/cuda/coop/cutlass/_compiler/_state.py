# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keep generated C++ requests separate for each active CuTe trace.

A DSL can reuse its compile-options object across nested or later traces, so
sessions are also identified by their MLIR module. Providers register requests
while emitting extern calls; finalization consumes only the matching session.
Snapshots support rollback of queued requests after a failed lowering.
"""

from __future__ import annotations

import importlib
import threading
import weakref
from collections.abc import Callable
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

from ._rendering import canonical_bundle_requests
from ._types import DeferredTempStorageEvent

_SESSION_SCOPE = "cuda.coop.cutlass"
_TRACE_HOOK_DISPATCHER_ATTR = (
    "_cuda_coop_cutlass_provider_trace_finalize_dispatcher"
)
_TRACE_HOOK_TARGET_ATTR = "_cuda_coop_cutlass_provider_trace_finalize_hook"
_BUNDLE_FINALIZER: Callable[[Any, Any, str], None] | None = None
_STATE_LOCK = threading.RLock()
_SESSIONS: weakref.WeakKeyDictionary[Any, list[BundleSession]] = (
    weakref.WeakKeyDictionary()
)
_ID_SESSIONS: dict[
    int,
    tuple[weakref.ReferenceType[Any], list[BundleSession], weakref.finalize],
] = {}
_UNSPECIFIED_MODULE = object()


class BundleSession:
    """Collect wrapper definitions and scratch uses for one CuTe trace.

    Wrapper requests form a set because identical calls share generated code.
    Scratch events retain every traced use in order: two calls to the same
    wrapper can require distinct exclusive slices. Snapshots capture both
    collections so a failed lowering cannot leave a stray scratch event. The
    lock protects collection and module changes; snapshots keep their own
    containers while sharing the immutable request descriptions.
    """

    def __init__(self, trace_module_op=None):
        self.trace_module_op = trace_module_op
        self.requests = set()
        self._deferred_temp_storage_events: list[DeferredTempStorageEvent] = []
        self._lock = threading.RLock()

    def add(self, request):
        with self._lock:
            self.requests.add(request)

    def snapshot(self):
        """Copy trace bookkeeping before a lowering step that can fail."""

        with self._lock:
            return (
                self.trace_module_op,
                set(self.requests),
                list(self._deferred_temp_storage_events),
            )

    def restore(self, snapshot):
        """Restore trace bookkeeping without changing emitted MLIR."""

        with self._lock:
            self.trace_module_op, requests, events = snapshot
            self.requests = set(requests)
            self._deferred_temp_storage_events = list(events)

    def request_list(self):
        """Order requests by symbol and reject conflicting definitions."""

        with self._lock:
            return list(canonical_bundle_requests(self.requests))

    def add_deferred_temp_storage_event(
        self, event: DeferredTempStorageEvent
    ) -> None:
        """Keep repeated calls even when they share one provider."""

        with self._lock:
            self._deferred_temp_storage_events.append(event)

    def deferred_temp_storage_event_list(
        self,
    ) -> list[DeferredTempStorageEvent]:
        with self._lock:
            return list(self._deferred_temp_storage_events)

    def is_empty(self):
        with self._lock:
            return not self.requests and not self._deferred_temp_storage_events

    def belongs_to_trace_module(self, module):
        with self._lock:
            return _same_mlir_operation(self.trace_module_op, module)

    def bind_trace_module(self, module):
        """Bind an unassigned session or check its existing module owner."""

        with self._lock:
            if self.trace_module_op is None:
                self.trace_module_op = module
            return self.belongs_to_trace_module(module)


def register_bundle_finalizer(
    finalizer: Callable[[Any, Any, str], None],
    *,
    scope: str = _SESSION_SCOPE,
) -> None:
    """Set the callback that compiles a completed trace session."""

    if not callable(finalizer):
        raise TypeError("finalizer must be callable")
    global _BUNDLE_FINALIZER, _SESSION_SCOPE
    with _STATE_LOCK:
        _BUNDLE_FINALIZER = finalizer
        _SESSION_SCOPE = scope


def _ensure_bundle_finalizer() -> Callable[[Any, Any, str], None]:
    """Load the finalizer lazily and require its registration side effect."""

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
    """Call the current finalizer through one persistent DSL hook."""

    hook = getattr(dsl, _TRACE_HOOK_TARGET_ATTR, None)
    if hook is not None:
        hook(dsl, module, function_name)


def ensure_trace_hook_registered(
    *,
    finalizer: Callable[[Any, Any, str], None] | None = None,
    scope: str | None = None,
    get_cute_dsl: Callable[[], Any] | None = None,
) -> None:
    """Install one DSL dispatcher and select its finalizer.

    The first cooperative request in a trace reaches this helper through
    ``active_bundle_session``. CuTe later calls the installed dispatcher
    once the module is traced, while its IR and link inputs can still be
    updated.

    Separate the stable registered hook from its replaceable target so
    repeated activation does not accumulate callbacks. Reject a runtime
    without the hook needed to attach generated device code before linking.

    Parameters
    ----------
    finalizer : callable or None
        Callback accepting the DSL, completed module, and function name.
        None loads the default provider finalizer lazily.
    scope : str or None
        Backend name for capability diagnostics. None keeps the current name.
    get_cute_dsl : callable or None
        Getter for the active DSL. None uses CuTe's current instance; callers
        can supply a getter when using another runtime context.
    """

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
    """Find sessions without retaining their compile-options owner.

    Unhashable options still need weak references. Check identity before
    reusing the fallback so a recycled Python object ID cannot inherit old
    requests.
    """

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
    """Select a matching module session or an unambiguous sole session.

    An unspecified module must not choose arbitrarily among nested traces.
    """

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
    """Find a trace session without removing other modules on the same DSL."""

    with _STATE_LOCK:
        return _select_session(
            _sessions_for_options(compile_options) or [], trace_module_op
        )


def _drop_id_session(key: int) -> None:
    with _STATE_LOCK:
        _ID_SESSIONS.pop(key, None)


def _store_bundle_sessions(
    compile_options: Any, sessions: list[BundleSession]
) -> None:
    """Store sessions without keeping their compile-options owner alive.

    Use an identity-keyed weak reference when options cannot be dictionary
    keys, and remove that entry when its owner is collected.
    """

    try:
        _SESSIONS[compile_options] = sessions
    except TypeError:
        try:
            options_ref = weakref.ref(compile_options)
        except TypeError as exc:
            raise DSLRuntimeError(
                f"{_SESSION_SCOPE} provider compile_options must be "
                "weak-referenceable."
            ) from exc
        key = id(compile_options)
        finalizer = weakref.finalize(compile_options, _drop_id_session, key)
        _ID_SESSIONS[key] = (options_ref, sessions, finalizer)


def set_bundle_session(compile_options: Any, session: BundleSession) -> None:
    """Replace one module session while retaining other trace sessions."""

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
    """Compare operation wrappers while handling failed equality checks."""

    lhs = getattr(lhs, "operation", lhs)
    rhs = getattr(rhs, "operation", rhs)
    if lhs is rhs:
        return True
    try:
        result = lhs == rhs
    except Exception:  # noqa: BLE001
        # Foreign MLIR wrappers may reject equality.
        return False
    return isinstance(result, bool) and result


def _active_trace_module_op() -> Any | None:
    """Walk the current insertion point to its enclosing builtin module.

    Unavailable insertion state means no active trace; no DSL or module is
    created as a fallback.
    """

    try:
        from cutlass._mlir import ir

        current_ip = ir.InsertionPoint.current
        op = None if current_ip is None else current_ip.block.owner
    except Exception:  # noqa: BLE001
        # No usable insertion point means no active trace.
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
    """Reuse this module's session, or bind an existing unbound session.

    Provider registration calls this while tracing; rollback can also use
    it to recreate a removed session. Compile options alone cannot identify
    a trace because CuTe can reuse them for nested or later compilations.

    Create and store a new session only when neither is available.

    Parameters
    ----------
    compile_options : object
        CuTe compile-options owner used to keep sessions alive only while
        that owner exists.
    trace_module_op : object or None
        MLIR module that owns the generated calls. None requests an unbound
        session; a later call with a module can bind that session to its
        trace.

    Returns
    -------
    BundleSession
        Existing or newly registered session for the requested owner and
        module.
    """

    with _STATE_LOCK:
        session = lookup_bundle_session(
            compile_options, trace_module_op=trace_module_op
        )
        if session is None and trace_module_op is not None:
            unbound = lookup_bundle_session(
                compile_options, trace_module_op=None
            )
            if unbound is not None:
                unbound.bind_trace_module(trace_module_op)
                session = unbound
        if session is None:
            session = BundleSession(trace_module_op=trace_module_op)
            set_bundle_session(compile_options, session)
        return session


def active_bundle_session() -> BundleSession:
    """Require an active module and prepare its session and finalize hook."""

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
    """Save active options, the module, requests, and scratch events.

    Lowerings take this snapshot before recording a wrapper request.
    If emission fails, ``restore_active_session_state_for`` removes those
    records so finalization does not compile providers for an abandoned call.

    A missing session is recorded explicitly so restoration can remove a
    session created by the failed operation.

    Parameters
    ----------
    get_cute_dsl : callable
        Getter for the DSL whose active module and compile options are saved.

    Returns
    -------
    tuple
        Compile-options owner, active module, and a copied session snapshot.
        The last entry is None when no session existed before the call.
    """

    compile_options = get_cute_dsl().compile_options
    module = _active_trace_module_op()
    with _STATE_LOCK:
        session = lookup_bundle_session(compile_options, trace_module_op=module)
        return (
            compile_options,
            module,
            None if session is None else session.snapshot(),
        )


def snapshot_active_session_state():
    return snapshot_active_session_state_for(get_cute_dsl=_get_cute_dsl)


def restore_active_session_state_for(
    snapshot,
    *,
    get_cute_dsl: Callable[[], Any],
) -> None:
    """Restore queued bundle state after a failed provider lowering.

    Remove a newly created session when the snapshot had none. If the active
    options changed, clear the current session before restoring the
    original. This restores request bookkeeping; it does not undo emitted
    MLIR or payload assignments.

    Parameters
    ----------
    snapshot : tuple or None
        Result of ``snapshot_active_session_state_for``. None discards the
        current module's session without restoring an earlier one.
    get_cute_dsl : callable
        Getter for the current DSL, used to find and clear a changed owner.
    """

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
        session = get_or_create_bundle_session(
            compile_options, trace_module_op=module
        )
        session.restore(session_snapshot)


def restore_active_session_state(snapshot) -> None:
    restore_active_session_state_for(snapshot, get_cute_dsl=_get_cute_dsl)


def register_request(request: Any) -> None:
    """Add an immutable provider request to the active trace's bundle."""

    active_bundle_session().add(request)


__all__ = [
    "_SESSION_SCOPE",
    "BundleSession",
    "active_bundle_session",
    "ensure_trace_hook_registered",
    "get_or_create_bundle_session",
    "lookup_bundle_session",
    "pop_bundle_session",
    "register_bundle_finalizer",
    "register_request",
    "restore_active_session_state",
    "restore_active_session_state_for",
    "set_bundle_session",
    "snapshot_active_session_state",
    "snapshot_active_session_state_for",
]

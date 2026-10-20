# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Connect common API markers to the compiler processing a kernel.

Importing an operation records its function object and supported group
kinds. Registration does not wrap the callable, so imported aliases are
the same object and keep its signature and documentation.

Numba recognizes registered function objects in its intermediate code. A
tracing compiler such as CuTe executes common API functions in Python, so
those functions ask registered probes which compiler environment is
active. Importing a backend makes its probe available; it does not select
that backend for every later call.

A separate context variable records which common operation is being
delegated. Qualified backend code uses this information to enforce common
API restrictions while retaining its own extensions for direct qualified
calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from types import ModuleType
from typing import Any, TypeVar

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup

_ACTIVE_COMMON_ROOT_OPERATION: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_common_root_operation",
    default=None,
)
_CallableT = TypeVar("_CallableT", bound=Callable[..., object])
_COMPILER_CONTEXT_PROBES: dict[str, Callable[[], bool]] = {}


@dataclass(frozen=True)
class _CommonGroupOperation:
    """Store the operation name and allowed groups for one public callable."""

    name: str
    group_kinds: tuple[str, ...]
    function: Callable[..., object]


_COMMON_GROUP_OPERATIONS_BY_NAME: dict[str, _CommonGroupOperation] = {}
_COMMON_GROUP_OPERATIONS_BY_FUNCTION: dict[
    Callable[..., object], _CommonGroupOperation
] = {}


def _common_group_operation(
    name: str,
    *,
    group_kinds: tuple[str, ...],
) -> Callable[[_CallableT], _CallableT]:
    """Record a public operation and the groups it accepts.

    The decorator runs when the API module is imported and records the
    function object without wrapping it. Compiler adapters use that object
    to recognize calls, including imported aliases. An unrelated function
    with the same name does not acquire this registration.
    """

    if not name or not group_kinds:
        raise ValueError(
            "common group operations require a name and group kinds"
        )

    def decorate(function: _CallableT) -> _CallableT:
        """Register one function in both tables and reject conflicts."""

        registration = _CommonGroupOperation(name, tuple(group_kinds), function)
        existing = _COMMON_GROUP_OPERATIONS_BY_NAME.get(name)
        if existing is not None and existing != registration:
            raise RuntimeError(
                f"common group operation {name!r} is already registered"
            )
        existing_function = _COMMON_GROUP_OPERATIONS_BY_FUNCTION.get(function)
        if existing_function is not None and existing_function != registration:
            raise RuntimeError(
                f"common group marker {function!r} is already registered"
            )
        _COMMON_GROUP_OPERATIONS_BY_NAME[name] = registration
        _COMMON_GROUP_OPERATIONS_BY_FUNCTION[function] = registration
        function.__cuda_coop_backend_member__ = name
        return function

    return decorate


def _common_group_operation_name(function: object) -> str | None:
    """Return the registered name used to recognize a common API call.

    The planner passes a resolved callable, including an imported alias.
    Return ``None`` when its identity has no operation registration.
    """

    registration = _COMMON_GROUP_OPERATIONS_BY_FUNCTION.get(function)
    return None if registration is None else registration.name


class UnsupportedCoopBackendOperationError(NotImplementedError):
    """Report that an operation is unavailable in the selected API or backend.

    Keep the module, operation, and stable reason code available to callers
    that need to classify the failure without parsing its display message.
    """

    def __init__(self, backend_module: str, operation: str) -> None:
        self.backend_module = backend_module
        self.operation = operation
        self.reason_code = "cuda-coop-backend-operation-unavailable"
        super().__init__(
            f"cuda.coop.{operation} is not implemented by {backend_module!r}"
        )


def _register_compiler_context_probe(
    backend_module: str, probe: Callable[[], bool]
) -> None:
    """Register a check for an initialized compiler environment.

    The backend supplies a callable that reports whether its environment is
    active now. A probe must inspect already initialized runtime objects
    without importing a compiler or creating a new environment. Registration
    replaces any earlier probe for the same backend module.
    """

    if not isinstance(backend_module, str) or not backend_module.strip():
        raise ValueError("backend_module must be a non-empty string")
    if not callable(probe):
        raise TypeError("compiler context probe must be callable")
    _COMPILER_CONTEXT_PROBES[backend_module] = probe


def _backend_module_name() -> str | None:
    """Find the single backend that owns the active compiler environment.

    Return None when no registered probe claims this trace. A failed probe or
    two positive probes raise a context error: choosing a fallback could send
    the same common call to the wrong compiler.
    """

    active = None
    for module_name, probe in tuple(_COMPILER_CONTEXT_PROBES.items()):
        try:
            owns_environment = probe()
        except Exception as exc:
            raise CoopCompilerContextRequiredError(
                f"Cannot inspect the compiler environment for {module_name!r}"
            ) from exc
        if owns_environment:
            if active is not None:
                raise CoopCompilerContextRequiredError(
                    f"Multiple backends own the current compiler environment: "
                    f"{active!r} and {module_name!r}"
                )
            active = module_name
    return active


@contextmanager
def _common_root_operation_scope(operation: str) -> Iterator[None]:
    """Mark a delegated common call while preserving nested call state.

    Backend code reads the operation name to apply common payload and dtype
    restrictions. The context variable is separate from backend selection and
    is restored even if validation or lowering raises an error.
    """

    token = _ACTIVE_COMMON_ROOT_OPERATION.set(operation)
    try:
        yield
    finally:
        _ACTIVE_COMMON_ROOT_OPERATION.reset(token)


def _common_root_operation_name() -> str | None:
    """Return the common-root operation currently delegated to a backend."""

    return _ACTIVE_COMMON_ROOT_OPERATION.get()


def _active_backend(feature: str) -> tuple[str, ModuleType]:
    """Import the backend named by the active environment probe.

    The probe identifies an initialized integration. Without an active
    compiler environment, report the requested feature in the context error
    instead of choosing a backend from installed packages.
    """

    module_name = _backend_module_name()
    if module_name is None:
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{feature} requires compiler-owned activation or a "
            "qualified backend import before compilation"
        )
    return module_name, import_module(module_name)


def _backend_member(name: str) -> Any:
    """Find an operation in the active backend or report it as unavailable.

    Translate a missing attribute into the common unsupported-operation
    error so callers get the backend and operation names instead of an
    AttributeError.
    """

    module_name, backend = _active_backend(name)
    try:
        return getattr(backend, name)
    except AttributeError as exc:
        raise UnsupportedCoopBackendOperationError(module_name, name) from exc


def _common_selector(
    operation: str,
    parameter: str,
    value: object,
    allowed: frozenset[str],
    *,
    allow_none: bool = False,
) -> Any:
    """Normalize a common selector during Python tracing.

    An active tracing backend receives a normalized string from the common
    allowlist. Reject Enum values and backend-only choices before delegation.
    Without an active probe, leave the value unchanged; a compiler that
    recognizes the original function marker performs its own planning checks.
    """

    if _backend_module_name() is None:
        return value
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or isinstance(value, Enum):
        raise TypeError(f"cuda.coop.{operation} {parameter} must be a string")
    token = value.strip().lower().replace("-", "_")
    try:
        is_allowed = token in allowed
    except TypeError:
        is_allowed = False
    if not is_allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"cuda.coop.{operation} {parameter} must be one of: {choices}; "
            "use a backend-qualified import for backend-only controls"
        )
    return token


def _common_group_name(kind: str) -> str:
    """Return the common API spelling for one internal group kind."""

    return "physical_warp" if kind == "warp" else kind


def _validate_common_operation_group(
    operation: str,
    group: ThreadGroup,
) -> None:
    """Reject groups outside a registered operation's common API contract.

    Compiler adapters call this after reconstructing a symbolic group
    descriptor and before lowering a recognized common operation. The registry
    defines the common set of group kinds; a backend-qualified operation can
    support more kinds without broadening that common contract.

    The descriptor may still have unresolved launch dimensions. This check
    validates only its type and group kind, not launch dimensions, primitive
    participation, or backend implementation.

    Parameters
    ----------
    operation : str
        Common operation name used to look up its registered group kinds and
        identify it in diagnostics.
    group : ThreadGroup
        Symbolic group descriptor whose ``kind`` must be supported by the
        registered operation. It is not modified.

    Raises
    ------
    TypeError
        ``group`` is not a ``ThreadGroup``.
    UnsupportedCoopBackendOperationError
        ``operation`` has no common operation registration.
    NotImplementedError
        The registered common operation does not support this group kind.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.{operation} group must be a ThreadGroup")
    registration = _COMMON_GROUP_OPERATIONS_BY_NAME.get(operation)
    if registration is None:
        raise UnsupportedCoopBackendOperationError("cuda.coop", operation)
    supported = registration.group_kinds
    if group.kind in supported:
        return
    group_name = _common_group_name(group.kind)
    supported_names = ", ".join(map(_common_group_name, supported))
    raise NotImplementedError(
        f"cuda.coop.{operation} does not support group kind {group_name!r} in "
        f"the common API; supported group kinds: {supported_names}; use a "
        "backend-qualified import for backend-specific group support"
    )


def _group_primitive_marker(
    operation: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch a registered common operation in an active Python trace.

    Check the common group contract before calling the backend. Keep the
    operation name in context for the duration of that call so qualified
    code can distinguish common restrictions from its direct-call
    extensions. Compilers that replace registered markers never run this
    path.
    """

    if operation not in _COMMON_GROUP_OPERATIONS_BY_NAME:
        raise UnsupportedCoopBackendOperationError("cuda.coop", operation)
    if _backend_module_name() is None:
        del args, kwargs
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{operation} requires compiler-owned activation or a "
            "qualified backend import before compilation"
        )
    _validate_common_operation_group(operation, args[0] if args else None)
    with _common_root_operation_scope(operation):
        return _backend_member(operation)(*args, **kwargs)


__all__ = [
    "_backend_member",
    "_backend_module_name",
    "_common_group_operation",
    "_common_group_operation_name",
    "_common_root_operation_name",
    "_common_root_operation_scope",
    "_common_selector",
    "_group_primitive_marker",
    "_register_compiler_context_probe",
    "_validate_common_operation_group",
]

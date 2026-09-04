# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable root-backend selection and dispatch state.

Compiler integrations activate one qualified backend through this module while
family frontends validate the portable profile before delegation. It owns no
primitive semantics, provider rendering, or compiler cache state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, TypeVar

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup

_ACTIVE_BACKEND_MODULE: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_backend_module",
    default=None,
)
_ACTIVE_COMMON_ROOT_OPERATION: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_common_root_operation",
    default=None,
)
_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


@dataclass(frozen=True)
class _PortableGroupOperation:
    name: str
    group_kinds: tuple[str, ...]
    function: Callable[..., Any]


_PORTABLE_GROUP_OPERATIONS_BY_NAME: dict[str, _PortableGroupOperation] = {}
_PORTABLE_GROUP_OPERATIONS_BY_FUNCTION: dict[
    Callable[..., Any], _PortableGroupOperation
] = {}


def _portable_group_operation(
    name: str,
    *,
    group_kinds: tuple[str, ...],
) -> Callable[[_CallableT], _CallableT]:
    """Register one portable group overload by exact callable identity."""

    if not name or not group_kinds:
        raise ValueError("portable group operations require a name and group kinds")

    def decorate(function: _CallableT) -> _CallableT:
        registration = _PortableGroupOperation(name, tuple(group_kinds), function)
        existing = _PORTABLE_GROUP_OPERATIONS_BY_NAME.get(name)
        if existing is not None and existing != registration:
            raise RuntimeError(
                f"portable group operation {name!r} is already registered"
            )
        existing_function = _PORTABLE_GROUP_OPERATIONS_BY_FUNCTION.get(function)
        if existing_function is not None and existing_function != registration:
            raise RuntimeError(
                f"portable group marker {function!r} is already registered"
            )
        _PORTABLE_GROUP_OPERATIONS_BY_NAME[name] = registration
        _PORTABLE_GROUP_OPERATIONS_BY_FUNCTION[function] = registration
        function.__cuda_coop_backend_member__ = name
        return function

    return decorate


def _portable_group_operation_name(function: Any) -> str | None:
    registration = _PORTABLE_GROUP_OPERATIONS_BY_FUNCTION.get(function)
    return None if registration is None else registration.name


class UnsupportedCoopBackendOperationError(NotImplementedError):
    """The selected compiler backend does not implement a root operation."""

    def __init__(self, backend_module: str, operation: str) -> None:
        self.backend_module = backend_module
        self.operation = operation
        self.reason_code = "cuda-coop-backend-operation-unavailable"
        super().__init__(
            f"cuda.coop.{operation} is not implemented by {backend_module!r}"
        )


@contextmanager
def _compiler_scope(backend_module: str) -> Iterator[None]:
    """Activate one backend for the current compiler trace."""

    if not isinstance(backend_module, str):
        raise TypeError("backend_module must be a string")
    if not backend_module.strip():
        raise ValueError("backend_module must be a non-empty string")

    token = _ACTIVE_BACKEND_MODULE.set(backend_module)
    try:
        yield
    finally:
        _ACTIVE_BACKEND_MODULE.reset(token)


def _backend_module_name() -> str | None:
    """Return the compiler-owned backend active in the current trace."""

    return _ACTIVE_BACKEND_MODULE.get()


@contextmanager
def _common_root_operation_scope(operation: str) -> Iterator[None]:
    """Identify one root dispatch without changing backend selection."""

    token = _ACTIVE_COMMON_ROOT_OPERATION.set(operation)
    try:
        yield
    finally:
        _ACTIVE_COMMON_ROOT_OPERATION.reset(token)


def _common_root_operation_name() -> str | None:
    """Return the common-root operation currently delegated to a backend."""

    return _ACTIVE_COMMON_ROOT_OPERATION.get()


def _active_backend(feature: str) -> tuple[str, ModuleType]:
    module_name = _backend_module_name()
    if module_name is None:
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{feature} requires compiler-owned activation or a "
            "qualified backend import before compilation"
        )
    return module_name, import_module(module_name)


def _backend_member(name: str) -> Any:
    module_name, backend = _active_backend(name)
    try:
        return getattr(backend, name)
    except AttributeError as exc:
        raise UnsupportedCoopBackendOperationError(module_name, name) from exc


def _portable_selector(
    operation: str,
    parameter: str,
    value: Any,
    allowed: frozenset[str],
    *,
    allow_none: bool = False,
) -> Any:
    """Normalize one common selector while a tracing backend is active."""

    if _backend_module_name() is None:
        return value
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
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


def _portable_group_name(kind: str) -> str:
    """Return the portable API spelling for one internal group kind."""

    return "physical_warp" if kind == "warp" else kind


def _validate_portable_operation_group(
    operation: str,
    group: Any,
) -> None:
    """Enforce the portable group matrix for a common-root call."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.{operation} group must be a ThreadGroup")
    registration = _PORTABLE_GROUP_OPERATIONS_BY_NAME.get(operation)
    if registration is None:
        raise UnsupportedCoopBackendOperationError("cuda.coop", operation)
    supported = registration.group_kinds
    if group.kind in supported:
        return
    group_name = _portable_group_name(group.kind)
    supported_names = ", ".join(map(_portable_group_name, supported))
    raise NotImplementedError(
        f"cuda.coop.{operation} does not support group kind {group_name!r} in "
        f"the portable API; supported group kinds: {supported_names}; use a "
        "backend-qualified import for backend-specific group support"
    )


def _group_primitive_marker(
    operation: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if operation not in _PORTABLE_GROUP_OPERATIONS_BY_NAME:
        raise UnsupportedCoopBackendOperationError("cuda.coop", operation)
    if _backend_module_name() is None:
        del args, kwargs
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{operation} requires compiler-owned activation or a "
            "qualified backend import before compilation"
        )
    _validate_portable_operation_group(operation, args[0] if args else None)
    with _common_root_operation_scope(operation):
        return _backend_member(operation)(*args, **kwargs)


__all__ = []

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
from importlib import import_module
from types import ModuleType
from typing import Any

from .._errors import CoopCompilerContextRequiredError, _BackendActivationFailure


class UnsupportedCoopBackendOperationError(NotImplementedError):
    """The active backend does not implement a common-root operation."""

    def __init__(self, backend_module: str, operation: str) -> None:
        self.backend_module = backend_module
        self.operation = operation
        self.reason_code = "cuda-coop-backend-operation-unavailable"
        super().__init__(
            f"cuda.coop.{operation} is not implemented by {backend_module!r}"
        )


_ACTIVE_BACKEND_MODULE: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_backend_module",
    default=None,
)
_ACTIVE_ROOT_OPERATION: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_root_operation",
    default=None,
)
_QUALIFIED_BACKEND_MODULE: str | None = None
_BACKEND_ACTIVATION_FAILURE: _BackendActivationFailure | None = None


def _validate_backend_module(backend_module: str) -> str:
    if not isinstance(backend_module, str):
        raise TypeError("backend_module must be a string")
    if not backend_module.strip():
        raise ValueError("backend_module must be a non-empty string")
    return backend_module


@contextmanager
def _compiler_scope(backend_module: str) -> Iterator[None]:
    """Activate one backend for the current compiler trace."""

    token = _ACTIVE_BACKEND_MODULE.set(_validate_backend_module(backend_module))
    try:
        yield
    finally:
        _ACTIVE_BACKEND_MODULE.reset(token)


def _register_qualified_backend(backend_module: str) -> None:
    """Activate one imported qualified backend for common-root calls."""

    backend_module = _validate_backend_module(backend_module)
    global _BACKEND_ACTIVATION_FAILURE, _QUALIFIED_BACKEND_MODULE
    if _QUALIFIED_BACKEND_MODULE not in {None, backend_module}:
        raise RuntimeError(
            "cuda.coop common-root fallback is already activated by "
            f"{_QUALIFIED_BACKEND_MODULE!r}; cannot also activate "
            f"{backend_module!r}"
        )
    _QUALIFIED_BACKEND_MODULE = backend_module
    _BACKEND_ACTIVATION_FAILURE = None


def _record_backend_activation_failure(
    backend: str,
    reason_code: str,
    cause: BaseException,
) -> None:
    """Retain optional-backend failure context for a later root call."""

    if not backend:
        raise ValueError("backend must be non-empty")
    if not reason_code:
        raise ValueError("reason_code must be non-empty")
    if not isinstance(cause, BaseException):
        raise TypeError("cause must be an exception")
    global _BACKEND_ACTIVATION_FAILURE
    _BACKEND_ACTIVATION_FAILURE = _BackendActivationFailure(
        backend=backend,
        reason_code=reason_code,
        cause=cause,
    )


def _backend_module_name() -> str | None:
    return _ACTIVE_BACKEND_MODULE.get() or _QUALIFIED_BACKEND_MODULE


@contextmanager
def _common_root_operation_scope(operation: str) -> Iterator[None]:
    """Identify one common-root dispatch for a compiler integration."""

    token = _ACTIVE_ROOT_OPERATION.set(operation)
    try:
        yield
    finally:
        _ACTIVE_ROOT_OPERATION.reset(token)


def _common_root_operation_name() -> str | None:
    """Return the common-root operation currently delegated to a backend."""

    return _ACTIVE_ROOT_OPERATION.get()


def _active_backend(feature: str) -> tuple[str, ModuleType]:
    module_name = _backend_module_name()
    if module_name is None:
        raise CoopCompilerContextRequiredError(feature, _BACKEND_ACTIVATION_FAILURE)
    return module_name, import_module(module_name)


def _backend_member(name: str) -> Any:
    module_name, backend = _active_backend(name)
    try:
        return getattr(backend, name)
    except AttributeError as error:
        raise UnsupportedCoopBackendOperationError(module_name, name) from error


def _group_primitive_marker(
    operation: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Delegate one validated root operation to the active backend."""

    with _common_root_operation_scope(operation):
        return _backend_member(operation)(*args, **kwargs)


__all__: list[str] = []

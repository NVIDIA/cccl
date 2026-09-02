# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral public API and qualified-backend dispatch."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
from types import ModuleType
from typing import Any, TypeVar

from .block.reduce import (
    normalize_block_reduce_algorithm,
    normalize_block_reduce_operator,
)
from .thread_group import ThreadGroup
from .thread_group import this_block as _core_this_block

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
_ScalarT = TypeVar("_ScalarT")


@dataclass(frozen=True)
class _BackendActivationFailure:
    backend: str
    reason_code: str
    cause: BaseException


class CoopCompilerContextRequiredError(RuntimeError):
    """A root reduction requires an active Python DSL backend."""

    def __init__(
        self,
        feature: str,
        activation_failure: _BackendActivationFailure | None = None,
    ) -> None:
        self.feature = feature
        self.cause = None if activation_failure is None else activation_failure.cause
        self.backend = (
            None if activation_failure is None else activation_failure.backend
        )
        self.reason_code = (
            "compiler-context-required"
            if activation_failure is None
            else activation_failure.reason_code
        )
        activation_details = (
            None if self.cause is None else getattr(self.cause, "details", None)
        )
        self.details = {
            "feature": feature,
            "backend": self.backend,
            "cause_type": None if self.cause is None else type(self.cause).__name__,
            "cause_message": None if self.cause is None else str(self.cause),
            "activation_details": activation_details,
        }
        message = (
            f"cuda.coop.{feature} requires an active compiler backend; "
            "install a compatible backend or import cuda.coop.numba_mlir "
            "before tracing a kernel"
        )
        if activation_failure is not None:
            message += (
                f"; automatic {activation_failure.backend} activation failed "
                f"({activation_failure.reason_code}): "
                f"{activation_failure.cause}"
            )
            self.__cause__ = activation_failure.cause
        super().__init__(message)


class UnsupportedCoopBackendOperationError(NotImplementedError):
    """The active backend does not implement a common-root operation."""

    def __init__(self, backend_module: str, operation: str) -> None:
        self.backend_module = backend_module
        self.operation = operation
        self.reason_code = "cuda-coop-backend-operation-unavailable"
        super().__init__(
            f"cuda.coop.{operation} is not implemented by {backend_module!r}"
        )


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


def _validate_block_group(group: ThreadGroup, *, operation: str) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.{operation} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"cuda.coop.{operation} currently supports block groups only"
        )


def _normalize_valid_items(operation: str, valid_items: Any) -> Any:
    if valid_items is None:
        return None
    if isinstance(valid_items, bool):
        raise TypeError(f"cuda.coop.{operation} valid_items must be an integer")
    if isinstance(valid_items, Integral):
        normalized = int(valid_items)
        if normalized < 1:
            raise ValueError(f"cuda.coop.{operation} valid_items must be at least 1")
        return normalized
    try:
        width = valid_items.width
        signed = valid_items.signed
        dtype = valid_items.dtype
        ir_value = valid_items.ir_value
    except AttributeError:
        pass
    else:
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(signed, bool)
            and dtype is not None
            and callable(ir_value)
        ):
            return valid_items
    raise TypeError(f"cuda.coop.{operation} valid_items must be an integer")


def this_block() -> ThreadGroup:
    """Return a descriptor for the current CUDA thread block.

    The descriptor carries no user-supplied dimensions. A compiler integration
    resolves exact block dimensions from verified launch facts when lowering a
    reduction.

    Returns:
        A compiler-free block descriptor accepted by ``reduce`` and ``sum``.

    Raises:
        RuntimeError: If a compiler later cannot resolve exact block dimensions.

    Example:
        >>> from cuda import coop
        >>> block = coop.this_block()
        >>> block.kind
        'block'
    """

    return _core_this_block()


def reduce(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: Any = None,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Reduce one scalar per block thread and return the root result.

    Every thread in ``group`` must participate in converged control flow. The
    return value is defined only for block rank zero; other threads must not
    consume it. ``valid_items`` selects a prefix of participating block ranks.

    Args:
        group: The current CUDA thread block.
        value: One numeric scalar owned by the calling thread.
        binary_op: Built-in reduction selector. The default is ``"sum"``.
        valid_items: Optional number of valid block ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector.

    Returns:
        The reduced scalar, defined only for block rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If a selector or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.reduce(block, value, binary_op="sum")
    """

    _validate_block_group(group, operation="reduce")
    operator = normalize_block_reduce_operator(binary_op)
    selected_algorithm = normalize_block_reduce_algorithm(algorithm)
    valid_items = _normalize_valid_items("reduce", valid_items)
    with _common_root_operation_scope("reduce"):
        return _backend_member("reduce")(
            group,
            value,
            binary_op=operator.value,
            valid_items=valid_items,
            algorithm=selected_algorithm.value,
        )


def sum(
    group: ThreadGroup,
    value: _ScalarT,
    /,
    *,
    valid_items: Any = None,
    algorithm: Any = None,
) -> _ScalarT:
    """Sum one scalar per block thread and return the root result.

    Every thread in ``group`` must participate in converged control flow. The
    return value is defined only for block rank zero; other threads must not
    consume it. ``valid_items`` selects a prefix of participating block ranks.

    Args:
        group: The current CUDA thread block.
        value: One numeric scalar owned by the calling thread.
        valid_items: Optional number of valid block ranks, starting at rank zero.
        algorithm: Optional deterministic CUB BlockReduce algorithm selector.

    Returns:
        The sum, defined only for block rank zero.

    Raises:
        TypeError: If ``group`` or a static ``valid_items`` is invalid.
        ValueError: If an algorithm or static ``valid_items`` is invalid.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        >>> total = coop.sum(block, value)
    """

    _validate_block_group(group, operation="sum")
    selected_algorithm = normalize_block_reduce_algorithm(algorithm)
    valid_items = _normalize_valid_items("sum", valid_items)
    with _common_root_operation_scope("sum"):
        return _backend_member("sum")(
            group,
            value,
            valid_items=valid_items,
            algorithm=selected_algorithm.value,
        )


for _member_name in ("this_block", "reduce", "sum"):
    globals()[_member_name].__cuda_coop_backend_member__ = _member_name
del _member_name


__all__ = ["ThreadGroup", "this_block", "reduce", "sum"]

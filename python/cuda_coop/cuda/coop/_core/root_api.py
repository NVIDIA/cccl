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
from types import ModuleType
from typing import Any, Protocol, TypeVar, runtime_checkable

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
_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ThreadDataLike(Protocol[_ItemT]):
    items_per_thread: int
    dtype: Any | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ItemT: ...

    def __setitem__(self, index: int, value: _ItemT) -> None: ...


@dataclass(frozen=True)
class _BackendActivationFailure:
    backend: str
    reason_code: str
    cause: BaseException


class CoopCompilerContextRequiredError(RuntimeError):
    """A common-root operation requires an active Python DSL backend."""

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
            "install or import a compatible backend before compiling a kernel"
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
    """Activate one backend while the compiler processes a kernel function."""

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
    token = _ACTIVE_ROOT_OPERATION.set(operation)
    try:
        yield
    finally:
        _ACTIVE_ROOT_OPERATION.reset(token)


def _common_root_operation_name() -> str | None:
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


def this_block() -> ThreadGroup:
    """Return a descriptor for the current CUDA thread block.

    The returned group has no user-supplied dimensions. The active backend
    supplies exact launch dimensions when it lowers a cooperative primitive.

    Returns:
        An opaque block descriptor accepted by cooperative primitives.

    Raises:
        RuntimeError: If a compiler backend later cannot resolve exact block
            dimensions for an operation using this descriptor.

    Example:
        This tested CUTLASS kernel uses the current CUDA thread block:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    return _core_this_block()


def ThreadData(items_per_thread: int, dtype: Any = None) -> _ThreadDataLike[Any]:
    """Create an uninitialized per-thread register payload.

    The active backend owns the concrete payload type. When ``dtype`` is
    omitted, a primitive may infer it from its inputs for use by later
    primitives.

    Args:
        items_per_thread: Number of consecutive values owned by each thread.
        dtype: Optional portable numeric dtype. A primitive may infer it from
            its inputs.

    Returns:
        The active compiler backend's fixed-size payload object.

    Raises:
        ValueError: If ``items_per_thread`` is not positive.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel creates and uses a per-thread payload:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread <= 0
    ):
        raise ValueError("items_per_thread must be a positive integer")
    with _common_root_operation_scope("ThreadData"):
        return _backend_member("ThreadData")(items_per_thread, dtype=dtype)


def load(
    group: ThreadGroup,
    source: Any,
    items: _ThreadDataLike[_ItemT],
    /,
    *,
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Collectively load one block tile into a per-thread payload.

    Every thread in ``group`` must participate in converged control flow. The
    payload size determines the number of consecutive values loaded per thread.
    Contiguous operands are traversed in linear storage order; multidimensional
    logical indexing is not applied.

    Args:
        group: The current CUDA thread block.
        source: Contiguous pointer-backed input memory.
        items: Payload whose size determines the values owned by each thread.
        valid_items: Optional valid element count for a partial block tile.
        oob_default: Optional value assigned to invalid Load positions.
        offset: Optional element offset from the input pointer.

    Returns:
        ``items`` after the active compiler backend populates it.

    Raises:
        TypeError: If ``group`` is invalid or ``oob_default`` is supplied
            without ``valid_items``.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel loads a partial block tile:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    _validate_block_group(group, operation="load")
    if oob_default is not None and valid_items is None:
        raise TypeError("cuda.coop.load oob_default requires valid_items")
    with _common_root_operation_scope("load"):
        return _backend_member("load")(
            group,
            source,
            items,
            valid_items=valid_items,
            oob_default=oob_default,
            offset=offset,
        )


def store(
    group: ThreadGroup,
    destination: Any,
    items: _ThreadDataLike[Any],
    /,
    *,
    valid_items: Any = None,
    offset: Any = None,
) -> None:
    """Collectively store one per-thread payload as one block tile.

    Every thread in ``group`` must participate in converged control flow. The
    payload size determines the number of consecutive values stored per thread.
    Contiguous operands are traversed in linear storage order; multidimensional
    logical indexing is not applied.

    Args:
        group: The current CUDA thread block.
        destination: Contiguous pointer-backed output memory.
        items: Fixed-size payload stored by each thread.
        valid_items: Optional valid element count for a partial block tile.
        offset: Optional element offset from the output pointer.

    Returns:
        ``None``.

    Raises:
        TypeError: If ``group`` is not a ``ThreadGroup``.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel stores a partial block tile:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    _validate_block_group(group, operation="store")
    with _common_root_operation_scope("store"):
        _backend_member("store")(
            group,
            destination,
            items,
            valid_items=valid_items,
            offset=offset,
        )


__all__ = ["ThreadData", "ThreadGroup", "this_block", "load", "store"]

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact callable-identity registries for planner-recognized operations.

Numba IR eventually exposes the Python callable assigned to a call site.  The
planner records and looks up those callable objects directly here; module and
function names are diagnostic metadata only and never establish identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any, Callable, TypeVar

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


@dataclass(frozen=True)
class FactoryOperation:
    """Semantic operation and execution scope for one lowering factory."""

    operation: str
    namespace: str


@dataclass(frozen=True)
class GroupPrimitiveRegistration:
    """Whole-function planning hooks owned by one primitive operation."""

    lower: Callable[..., list[Any]]
    array_result_parameter: str | None = None
    validate_common_arguments: Callable[..., None] | None = None


@dataclass(frozen=True)
class RewriteOperationSpec:
    """Before-inference ABI and family hooks for one provider operation."""

    namespace: str
    runtime_arg_counts: frozenset[int]
    runtime_factory_kwargs: tuple[str, ...]
    runtime_factory_kw_prerequisites: tuple[tuple[str, str], ...]
    allowed_factory_kwargs: frozenset[str]
    required_factory_kwargs: frozenset[str]
    runtime_temp_storage: bool
    scalar_binding_kwargs: frozenset[str]
    runtime_offset_kwarg: str | None
    infer_payload: Callable[[Any, Any], None]
    analyze_match: Callable[..., Any] | None = None
    prepare_runtime_args: Callable[..., list[Any]] | None = None
    validate_runtime_controls: Callable[..., None] | None = None


_GROUP_OPERATIONS: dict[Callable[..., Any], str] = {}
_GROUP_FAMILY_MODULES: dict[str, str] = {}
_FACTORY_OPERATIONS: dict[Callable[..., Any], FactoryOperation] = {}
_GROUP_PRIMITIVES: dict[str, GroupPrimitiveRegistration] = {}
_REWRITE_OPERATIONS: dict[str, RewriteOperationSpec] = {}
_GROUP_FAMILY_IMPORT_LOCK = RLock()


def group_operation(
    operation: str,
    *,
    family_module: str,
) -> Callable[[_CallableT], _CallableT]:
    """Register one public group marker by exact callable identity."""

    def decorate(function: _CallableT) -> _CallableT:
        existing = _GROUP_OPERATIONS.get(function)
        if existing is not None and existing != operation:
            raise RuntimeError(
                f"group marker {function!r} is already registered as {existing!r}"
            )
        _GROUP_OPERATIONS[function] = operation
        existing_module = _GROUP_FAMILY_MODULES.get(operation)
        if existing_module is not None and existing_module != family_module:
            raise RuntimeError(
                f"group operation {operation!r} is already assigned to "
                f"compiler family {existing_module!r}"
            )
        _GROUP_FAMILY_MODULES[operation] = family_module
        function.__cuda_coop_backend_member__ = operation
        return function

    return decorate


def group_operation_name(function: Any) -> str | None:
    """Return the operation for an exactly registered group marker."""

    return _GROUP_OPERATIONS.get(function)


def _ensure_group_family_loaded(operation: str) -> None:
    module_name = _GROUP_FAMILY_MODULES.get(operation)
    if module_name is None:
        backend = import_module("cuda.coop.numba_mlir")
        getattr(backend, operation, None)
        module_name = _GROUP_FAMILY_MODULES.get(operation)
        if module_name is None:
            return
    with _GROUP_FAMILY_IMPORT_LOCK:
        import_module(module_name)


def register_group_primitive(
    operation: str,
    *,
    lower: Callable[..., list[Any]],
    array_result_parameter: str | None = None,
    validate_common_arguments: Callable[..., None] | None = None,
) -> None:
    """Register the post-inlining planner for one public operation."""

    registration = GroupPrimitiveRegistration(
        lower=lower,
        array_result_parameter=array_result_parameter,
        validate_common_arguments=validate_common_arguments,
    )
    existing = _GROUP_PRIMITIVES.get(operation)
    if existing is not None and existing != registration:
        raise RuntimeError(f"group primitive {operation!r} is already registered")
    _GROUP_PRIMITIVES[operation] = registration


def group_primitive(operation: str) -> GroupPrimitiveRegistration | None:
    """Return the planning hooks registered for an operation name."""

    if operation not in _GROUP_PRIMITIVES:
        _ensure_group_family_loaded(operation)
    return _GROUP_PRIMITIVES.get(operation)


def register_rewrite_operation(
    operation: str,
    spec: RewriteOperationSpec,
) -> None:
    """Register one provider ABI with the shared before-inference rewrite."""

    if not isinstance(spec, RewriteOperationSpec):
        raise TypeError("spec must be a RewriteOperationSpec")
    existing = _REWRITE_OPERATIONS.get(operation)
    if existing is not None and existing != spec:
        raise RuntimeError(f"rewrite operation {operation!r} is already registered")
    _REWRITE_OPERATIONS[operation] = spec


def rewrite_operation(operation: str) -> RewriteOperationSpec | None:
    """Return the before-inference registration for one operation."""

    if operation not in _REWRITE_OPERATIONS:
        _ensure_group_family_loaded(operation)
    return _REWRITE_OPERATIONS.get(operation)


def register_factory(
    function: _CallableT,
    *,
    operation: str,
    namespace: str,
) -> _CallableT:
    """Register a primitive provider without relying on its import path."""

    metadata = FactoryOperation(operation=operation, namespace=namespace)
    existing = _FACTORY_OPERATIONS.get(function)
    if existing is not None and existing != metadata:
        raise RuntimeError(
            f"lowering factory {function!r} is already registered as {existing!r}"
        )
    _FACTORY_OPERATIONS[function] = metadata
    return function


def factory_operation(function: Any) -> FactoryOperation | None:
    """Return metadata for an exactly registered lowering factory."""

    return _FACTORY_OPERATIONS.get(function)


__all__ = [
    "FactoryOperation",
    "GroupPrimitiveRegistration",
    "RewriteOperationSpec",
    "factory_operation",
    "group_primitive",
    "group_operation",
    "group_operation_name",
    "register_factory",
    "register_group_primitive",
    "register_rewrite_operation",
    "rewrite_operation",
]

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Connect public cooperative calls to their compiler implementations.

A group marker is a public callable such as ``load`` that the group planner
recognizes in kernel IR. A lowering factory is a host-side callable, such as
``_lowering._load_store._load_with_storage``, that accepts specialization
inputs (dtype, block dimensions, algorithm, and scalar bindings) and builds
the compiled provider for that operation. Its usual result is an
``Invocable``: a callable carrying link artifacts and temporary-storage
requirements. During batch collection it returns an uncompiled ``Algorithm``
specialization instead.

These registries identify markers and factories by callable identity, so
aliases work without relying on function names. Operation names connect
those identities to family-specific group lowering, argument validation,
payload inference, and runtime-argument preparation. Families load lazily
when their hooks are needed; registration itself does not compile providers
or run cooperative operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from ._group_rewriting import GroupRewriteContext
    from ._rewrite_payload import PayloadInference

from cuda.coop._core import SynchronizationScope

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])

# The whole-function group planner attaches its exact lowering record to the
# provider marker with this reserved keyword.  The provider rewrite
# consumes it before invoking the registered provider factory.
_GROUP_LOWERING_PLAN_KWARG = "__cuda_coop_group_lowering_plan__"


class _InferPayloadHook(Protocol):
    def __call__(
        self,
        context: GroupRewriteContext,
        inference: PayloadInference,
    ) -> None: ...


class _AnalyzeMatchHook(Protocol):
    def __call__(
        self,
        context: GroupRewriteContext,
        *,
        op_name: str,
        runtime_args: tuple[Any, ...],
        factory_kwargs: dict[str, object],
    ) -> Any: ...


class _PrepareRuntimeArgsHook(Protocol):
    def __call__(
        self,
        context: GroupRewriteContext,
        block: Any,
        *,
        match: Any,
        runtime_args: list[Any],
        scope: Any,
        loc: Any,
    ) -> list[Any]: ...


class _ValidateRuntimeControlsHook(Protocol):
    def __call__(
        self,
        context: GroupRewriteContext,
        *,
        op_name: str,
        runtime_args: list[Any],
        factory_kwargs: dict[str, object],
    ) -> None: ...


class StorageABI(str, Enum):
    """How a generated provider receives temporary storage."""

    NONE = "none"
    LEADING_POINTER = "leading_pointer"


@dataclass(frozen=True)
class FactoryOperation:
    """Declarative contract for one registered lowering factory."""

    operation: str
    namespace: str
    storage_abi: StorageABI
    execution_scope: SynchronizationScope
    synchronization_scope: SynchronizationScope

    def __post_init__(self) -> None:
        for name in ("operation", "namespace"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "storage_abi", StorageABI(self.storage_abi))
        object.__setattr__(
            self,
            "execution_scope",
            SynchronizationScope(self.execution_scope),
        )
        object.__setattr__(
            self,
            "synchronization_scope",
            SynchronizationScope(self.synchronization_scope),
        )
        if self.synchronization_scope not in {
            SynchronizationScope.NONE,
            self.execution_scope,
        }:
            raise ValueError(
                "synchronization_scope must be NONE or match execution_scope"
            )


@dataclass(frozen=True)
class GroupPrimitiveRegistration:
    """Whole-function planning hooks owned by one primitive operation."""

    lower: Callable[..., list[Any]]
    validate_common_arguments: Callable[..., None] | None = None

    def __post_init__(self) -> None:
        if not callable(self.lower):
            raise TypeError("lower must be callable")
        if self.validate_common_arguments is not None and not callable(
            self.validate_common_arguments
        ):
            raise TypeError(
                "validate_common_arguments must be callable or None"
            )


@dataclass(frozen=True)
class RewriteOperationSpec:
    """Before-inference call grammar and hooks for one operation."""

    factory_namespaces: frozenset[str]
    dtype_factory_kwargs: frozenset[str]
    runtime_arg_counts: frozenset[int]
    runtime_factory_kwargs: tuple[str, ...]
    runtime_factory_kw_prerequisites: tuple[tuple[str, str], ...]
    allowed_factory_kwargs: frozenset[str]
    required_factory_kwargs: frozenset[str]
    accepts_temp_storage: bool
    scalar_binding_kwargs: frozenset[str]
    runtime_offset_kwarg: str | None
    infer_payload: _InferPayloadHook
    analyze_match: _AnalyzeMatchHook | None = None
    prepare_runtime_args: _PrepareRuntimeArgsHook | None = None
    validate_runtime_controls: _ValidateRuntimeControlsHook | None = None

    def __post_init__(self) -> None:
        for name in (
            "factory_namespaces",
            "dtype_factory_kwargs",
            "runtime_arg_counts",
            "allowed_factory_kwargs",
            "required_factory_kwargs",
            "scalar_binding_kwargs",
        ):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        object.__setattr__(
            self,
            "runtime_factory_kwargs",
            tuple(self.runtime_factory_kwargs),
        )
        raw_prerequisites = tuple(self.runtime_factory_kw_prerequisites)
        prerequisites: list[tuple[str, str]] = []
        for prerequisite in raw_prerequisites:
            if (
                not isinstance(prerequisite, (tuple, list))
                or len(prerequisite) != 2
            ):
                raise TypeError(
                    "runtime_factory_kw_prerequisites must contain name pairs"
                )
            name, required_name = prerequisite
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "runtime_factory_kw_prerequisite "
                    "names must be non-empty strings"
                )
            if not isinstance(required_name, str) or not required_name:
                raise ValueError(
                    "runtime_factory_kw_prerequisite "
                    "names must be non-empty strings"
                )
            prerequisites.append((name, required_name))
        object.__setattr__(
            self,
            "runtime_factory_kw_prerequisites",
            tuple(prerequisites),
        )
        if not self.factory_namespaces or any(
            not isinstance(namespace, str) or not namespace
            for namespace in self.factory_namespaces
        ):
            raise ValueError(
                "factory_namespaces must contain non-empty strings"
            )
        for field_name in (
            "dtype_factory_kwargs",
            "allowed_factory_kwargs",
            "required_factory_kwargs",
            "scalar_binding_kwargs",
        ):
            if any(
                not isinstance(name, str) or not name
                for name in getattr(self, field_name)
            ):
                raise ValueError(f"{field_name} must contain non-empty strings")
        if not self.runtime_arg_counts:
            raise ValueError("runtime_arg_counts must not be empty")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in self.runtime_arg_counts
        ):
            raise ValueError(
                "runtime_arg_counts must contain non-negative integers"
            )
        if any(
            not isinstance(name, str) or not name
            for name in self.runtime_factory_kwargs
        ):
            raise ValueError(
                "runtime_factory_kwargs must contain non-empty strings"
            )
        if len(set(self.runtime_factory_kwargs)) != len(
            self.runtime_factory_kwargs
        ):
            raise ValueError("runtime_factory_kwargs must be unique")
        runtime_factory_kwargs = frozenset(self.runtime_factory_kwargs)
        unknown_runtime_kwargs = (
            runtime_factory_kwargs - self.allowed_factory_kwargs
        )
        if unknown_runtime_kwargs:
            names = ", ".join(sorted(unknown_runtime_kwargs))
            raise ValueError(
                f"runtime_factory_kwargs must be "
                f"allowed factory kwargs: {names}"
            )
        base_runtime_arg_count = min(self.runtime_arg_counts)
        if max(self.runtime_arg_counts) - base_runtime_arg_count > len(
            self.runtime_factory_kwargs
        ):
            raise ValueError(
                "runtime_arg_counts require more "
                "trailing runtime arguments than "
                "runtime_factory_kwargs declares"
            )
        prerequisite_names = [name for name, _ in prerequisites]
        if len(set(prerequisite_names)) != len(prerequisite_names):
            raise ValueError(
                "runtime_factory_kw_prerequisite names must be unique"
            )
        known_prerequisites = (
            runtime_factory_kwargs | self.allowed_factory_kwargs
        )
        for name, required_name in prerequisites:
            if name not in runtime_factory_kwargs:
                raise ValueError(
                    "runtime_factory_kw_prerequisite targets must be runtime "
                    f"factory kwargs: {name}"
                )
            if required_name not in known_prerequisites:
                raise ValueError(
                    "runtime_factory_kw_prerequisite "
                    "requirements must be known "
                    f"factory kwargs: {required_name}"
                )
            if name == required_name:
                raise ValueError(
                    "runtime_factory_kw_prerequisites cannot require themselves"
                )
        unknown_dtype_kwargs = (
            self.dtype_factory_kwargs - self.allowed_factory_kwargs
        )
        if unknown_dtype_kwargs:
            names = ", ".join(sorted(unknown_dtype_kwargs))
            raise ValueError(
                f"dtype_factory_kwargs must be allowed factory kwargs: {names}"
            )
        unknown_required_kwargs = (
            self.required_factory_kwargs - self.allowed_factory_kwargs
        )
        if unknown_required_kwargs:
            names = ", ".join(sorted(unknown_required_kwargs))
            raise ValueError(
                f"required_factory_kwargs must be "
                f"allowed factory kwargs: {names}"
            )
        unknown_scalar_kwargs = (
            self.scalar_binding_kwargs - runtime_factory_kwargs
        )
        if unknown_scalar_kwargs:
            names = ", ".join(sorted(unknown_scalar_kwargs))
            raise ValueError(
                f"scalar_binding_kwargs must be runtime factory kwargs: {names}"
            )
        if self.runtime_offset_kwarg is not None:
            if (
                not isinstance(self.runtime_offset_kwarg, str)
                or not self.runtime_offset_kwarg
            ):
                raise ValueError(
                    "runtime_offset_kwarg must be a non-empty string or None"
                )
            if self.runtime_offset_kwarg not in self.allowed_factory_kwargs:
                raise ValueError(
                    "runtime_offset_kwarg must be an allowed factory kwarg"
                )
            if self.runtime_offset_kwarg in runtime_factory_kwargs:
                raise ValueError(
                    "runtime_offset_kwarg must not "
                    "also be a runtime factory kwarg"
                )
        if not isinstance(self.accepts_temp_storage, bool):
            raise TypeError("accepts_temp_storage must be a bool")
        if not callable(self.infer_payload):
            raise TypeError("infer_payload must be callable")
        for name in (
            "analyze_match",
            "prepare_runtime_args",
            "validate_runtime_controls",
        ):
            hook = getattr(self, name)
            if hook is not None and not callable(hook):
                raise TypeError(f"{name} must be callable or None")


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
    """Associate a public group marker with its compiler family.

    This decorator records the exact callable object, so the planner recognizes
    aliases of the registered function without treating unrelated functions with
    the same name as cooperative operations. Record the family module for lazy
    loading of planning/rewrite hooks and attach the backend-member marker used
    during common API provenance checks. Registration does not import that
    family or wrap the decorated function.

    Parameters
    ----------
    operation : str
        Shared operation identifier used by group planning and rewrite lookup.
    family_module : str
        Importable compiler-family module that registers the operation's hooks.

    Returns
    -------
    callable
        Decorator that mutates the registries and callable metadata, then
        returns the original function. Repeating the same registration is
        allowed.

    Raises
    ------
    RuntimeError
        Applying the decorator would associate an already registered callable
        with another operation, or an operation with another family module.
    """

    def decorate(function: _CallableT) -> _CallableT:
        existing = _GROUP_OPERATIONS.get(function)
        if existing is not None and existing != operation:
            raise RuntimeError(
                f"group marker {function!r} is "
                f"already registered as {existing!r}"
            )
        _GROUP_OPERATIONS[function] = operation
        existing_module = _GROUP_FAMILY_MODULES.get(operation)
        if existing_module is not None and existing_module != family_module:
            raise RuntimeError(
                f"group operation {operation!r} is already assigned to "
                f"compiler family {existing_module!r}"
            )
        _GROUP_FAMILY_MODULES[operation] = family_module
        function.__dict__["__cuda_coop_backend_member__"] = operation
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
    validate_common_arguments: Callable[..., None] | None = None,
) -> None:
    """Register the post-inlining planner for one public operation."""

    registration = GroupPrimitiveRegistration(
        lower=lower,
        validate_common_arguments=validate_common_arguments,
    )
    existing = _GROUP_PRIMITIVES.get(operation)
    if existing is not None and existing != registration:
        raise RuntimeError(
            f"group primitive {operation!r} is already registered"
        )
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
    """Register one provider ABI with the shared before-inference rewrite.

    Re-registering an equal specification is allowed; a different specification
    for the same operation would make call interpretation ambiguous.

    Parameters
    ----------
    operation : str
        Operation name used to look up the call grammar and rewrite hooks.
    spec : RewriteOperationSpec
        Provider call grammar and hooks for this operation.

    Raises
    ------
    TypeError
        ``spec`` is not a ``RewriteOperationSpec``.
    RuntimeError
        The operation is already registered with a different specification.
    """

    if not isinstance(spec, RewriteOperationSpec):
        raise TypeError("spec must be a RewriteOperationSpec")
    existing = _REWRITE_OPERATIONS.get(operation)
    if existing is not None and existing != spec:
        raise RuntimeError(
            f"rewrite operation {operation!r} is already registered"
        )
    _REWRITE_OPERATIONS[operation] = spec


def rewrite_operation(operation: str) -> RewriteOperationSpec | None:
    """Return the before-inference registration for one operation.

    Load the owning primitive family when its registration is missing,
    without eagerly importing every family.

    Parameters
    ----------
    operation : str
        Operation name whose call grammar and rewrite hooks are requested.

    Returns
    -------
    RewriteOperationSpec or None
        Registered specification, or ``None`` if no registration exists after
        loading the owning family.
    """

    if operation not in _REWRITE_OPERATIONS:
        _ensure_group_family_loaded(operation)
    return _REWRITE_OPERATIONS.get(operation)


def register_factory(
    function: _CallableT,
    *,
    operation: str,
    namespace: str,
    storage_abi: StorageABI,
    execution_scope: SynchronizationScope,
    synchronization_scope: SynchronizationScope,
) -> _CallableT:
    """Register a provider-building function and its storage and thread scopes.

    The provider rewrite identifies factories by object identity and
    uses this metadata to validate provider calls and materialized invocables.
    The factory's import path or function name is not used to infer its ABI.
    Registration only declares metadata; the factory and source emitter must
    implement the declared storage and synchronization behavior.

    Parameters
    ----------
    function : callable
        Host-side callable that builds an ``Invocable`` from specialization
        keywords, or returns an ``Algorithm`` during batch collection. See the
        module overview for the distinction from public group markers.
    operation : str
        Non-empty operation identifier, such as ``"load"`` or ``"store"``.
    namespace : str
        Non-empty provider namespace, such as ``"block"`` or ``"warp"``.
    storage_abi : StorageABI
        Whether provider calls have a leading scratch pointer or no scratch.
    execution_scope : SynchronizationScope
        Scope of threads executing the cooperative operation.
    synchronization_scope : SynchronizationScope
        Declared synchronization scope, either ``NONE`` or the execution scope.

    Returns
    -------
    callable
        The original factory after registry insertion. Identical repeated
        registration is accepted.

    Raises
    ------
    TypeError
        ``function`` is not callable.
    ValueError
        Names, enum values, or the relationship between scopes are invalid.
    RuntimeError
        This exact factory is already registered with different metadata.
    """

    if not callable(function):
        raise TypeError("lowering factory must be callable")
    metadata = FactoryOperation(
        operation=operation,
        namespace=namespace,
        storage_abi=storage_abi,
        execution_scope=execution_scope,
        synchronization_scope=synchronization_scope,
    )
    existing = _FACTORY_OPERATIONS.get(function)
    if existing is not None and existing != metadata:
        raise RuntimeError(
            f"lowering factory {function!r} is "
            f"already registered as {existing!r}"
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
    "StorageABI",
    "factory_operation",
    "group_operation",
    "group_operation_name",
    "group_primitive",
    "register_factory",
    "register_group_primitive",
    "register_rewrite_operation",
    "rewrite_operation",
]

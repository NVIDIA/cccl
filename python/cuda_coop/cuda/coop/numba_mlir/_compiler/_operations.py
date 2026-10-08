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
aliases work without relying on function names. Operation names connect those
identities to family-specific group lowering, argument validation, payload
inference, and runtime-argument preparation. Families load lazily when their
hooks are needed; registration itself does not compile providers or run
cooperative operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from cuda.coop._core import (
        GroupTopologyRequirements,
        SynchronizationRequirements,
        TempStorageRequirements,
    )

    from ._group_rewriting import GroupRewriteContext
    from ._rewrite_payload import PayloadInference

from cuda.coop._core import StorageOwnership, SynchronizationScope

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])

# The whole-function group planner attaches its exact lowering record to the
# provider marker with this reserved keyword.  The provider rewrite
# consumes it before invoking the registered provider factory.
_GROUP_LOWERING_PLAN_KWARG = "__cuda_coop_group_lowering_plan__"


class _InferPayloadHook(Protocol):
    """Complete factory inputs from planned or inferred payload facts.

    Update the supplied inference state with compatible dtype and shape facts.
    That state carries the call's optional lowering plan. The context supplies
    the active rewrite's operand and constructor analysis.
    """

    def __call__(
        self,
        context: GroupRewriteContext,
        inference: PayloadInference,
    ) -> None: ...


class _AnalyzeMatchHook(Protocol):
    """Record family facts needed when replacing a provider call.

    Inspect the split arguments, consume private factory markers, and return
    metadata for the later runtime-argument hook.
    """

    def __call__(
        self,
        context: GroupRewriteContext,
        *,
        op_name: str,
        runtime_args: tuple[Any, ...],
        factory_kwargs: dict[str, object],
    ) -> Any: ...


class _PrepareRuntimeArgsHook(Protocol):
    """Prepare operands while the rewrite emits a provider call.

    Use the match's family metadata, append preparation IR to ``block``, and
    return the provider operands. ``scope`` and ``loc`` identify the generated
    variables and statements.
    """

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
    """Check operation-specific scalar controls before provider creation.

    Use available operand types and factory bindings. The hook may leave
    unknown types for the later compiler typing pass.
    """

    def __call__(
        self,
        context: GroupRewriteContext,
        *,
        op_name: str,
        runtime_args: list[Any],
        factory_kwargs: dict[str, object],
    ) -> None: ...


class StorageABI(str, Enum):
    """Describe the provider's temporary-storage calling convention.

    ``NONE`` needs no storage operand. ``LEADING_POINTER`` reserves the first
    provider argument for scratch memory, supplied separately from the
    operation's ordinary operands.
    """

    NONE = "none"
    LEADING_POINTER = "leading_pointer"


@dataclass(frozen=True)
class FactoryOperation:
    """Declare the call contract of a registered provider factory.

    The rewrite uses this record to select argument rules and storage
    handling, then checks the resulting invocable against it. Registration
    declares the contract; it does not inspect generated device code.

    Attributes
    ----------
    operation : str
        Operation identifier used to find its argument and rewrite rules.
    namespace : str
        Provider namespace, such as ``"block"`` or ``"warp"``.
    storage_abi : StorageABI
        Whether the generated call receives a leading scratch pointer.
    execution_scope : SynchronizationScope
        Threads that execute the primitive together.
    synchronization_scope : SynchronizationScope
        Declared synchronization scope: ``NONE`` or the execution scope.
        Storage planning decides which reuse barriers to emit.
    """

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


def expected_storage_reuse_barrier(
    topology: GroupTopologyRequirements,
    storage: TempStorageRequirements,
) -> SynchronizationScope:
    """Require a reuse barrier only for automatically synchronized storage."""

    return (
        topology.execution_scope
        if storage.ownership is not StorageOwnership.NONE and storage.auto_sync
        else SynchronizationScope.NONE
    )


def provider_synchronization_matches(
    metadata: FactoryOperation,
    topology: GroupTopologyRequirements,
    synchronization: SynchronizationRequirements,
    storage: TempStorageRequirements,
) -> bool:
    """Check the provider's barrier declaration against planned scratch reuse.

    Caller-owned storage may retain its allocating wrapper's declaration when
    automatic synchronization is disabled: pointer rewrites bypass that wrapper
    and let the caller synchronize. Implementation-owned storage has no such
    exception. Callers validate the planned reuse barrier separately.
    """

    planned_barrier = synchronization.storage_reuse_barrier
    return metadata.synchronization_scope is planned_barrier or (
        planned_barrier is SynchronizationScope.NONE
        and storage.ownership is StorageOwnership.CALLER
        and not storage.auto_sync
        and metadata.synchronization_scope is topology.execution_scope
    )


@dataclass(frozen=True)
class GroupResultSource:
    """Describe one public result's dtype and scalar or array shape.

    Group planning reads these policies before a call is lowered. This lets a
    later group call inspect an earlier call's result before normal Numba
    type inference. Names refer to bound public call arguments, not positions
    in the private provider ABI. Dtype and shape can come from different
    sources: ranks use int32 and retain the key payload's shape.

    Attributes
    ----------
    dtype_parameter : str or None
        Argument whose dtype the result inherits when neither an explicit
        dtype keyword nor a fixed dtype supplies it. ``None`` supplies no
        argument-based dtype inference.
    array_parameter : str or None
        Argument whose scalar/array form and array extent the result inherits
        when no extent resolver is present. ``None`` then describes one
        scalar item.
    fixed_dtype : object, optional
        Compiler dtype used when the dtype keyword supplies no value.
        ``None`` leaves inference to the named argument, if one exists.
    dtype_keyword : str or None
        Argument whose compile-time dtype, when non-None, takes precedence
        over the fixed dtype and source argument.
    extent_resolver : callable or None
        Hook that receives the planning context and bound call and returns a
        known per-thread extent or ``None``. Its presence declares an array
        result independently of input shape.
    """

    dtype_parameter: str | None
    array_parameter: str | None
    fixed_dtype: Any = None
    dtype_keyword: str | None = None
    extent_resolver: Callable[[Any, Any], int | None] | None = None

    def __post_init__(self) -> None:
        for name in ("dtype_parameter", "array_parameter", "dtype_keyword"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.extent_resolver is not None and not callable(
            self.extent_resolver
        ):
            raise TypeError("extent_resolver must be callable or None")


@dataclass(frozen=True)
class GroupPrimitiveRegistration:
    """Connect a public operation to its planner and result policies.

    ``lower`` returns replacement IR after the planner binds public arguments
    and resolves the group. ``validate_common_arguments`` checks calls through
    the common ``cuda.coop`` API first; backend-qualified calls skip it.
    ``None`` omits that check. Both hooks receive the active planning context.

    ``results`` describes return values in order so dtype and extent analysis
    can follow a call before rewriting. An optional ``result_resolver`` uses
    the planning context and bound arguments to replace that fixed tuple.
    This lets a static selector choose the result layout. One result is
    returned directly; multiple result policies describe a tuple. An empty
    tuple supplies no result provenance. These policies aid inference; they
    do not allocate or validate returned values.
    """

    lower: Callable[..., list[Any]]
    results: tuple[GroupResultSource, ...] = ()
    validate_common_arguments: Callable[..., None] | None = None
    result_resolver: (
        Callable[[Any, Any], tuple[GroupResultSource, ...]] | None
    ) = None

    def __post_init__(self) -> None:
        if not callable(self.lower):
            raise TypeError("lower must be callable")
        object.__setattr__(self, "results", tuple(self.results))
        if any(
            not isinstance(result, GroupResultSource) for result in self.results
        ):
            raise TypeError("results must contain GroupResultSource records")
        if self.result_resolver is not None and not callable(
            self.result_resolver
        ):
            raise TypeError("result_resolver must be callable or None")
        if self.validate_common_arguments is not None and not callable(
            self.validate_common_arguments
        ):
            raise TypeError(
                "validate_common_arguments must be callable or None"
            )


@dataclass(frozen=True)
class RewriteOperationSpecification:
    """Define how to interpret and replace one provider call.

    Public group planning first selects a provider factory. This record then
    separates its compile-time inputs from device operands and selects the
    family hooks used before compiler type inference. A registration can serve
    several provider namespaces.

    Attributes
    ----------
    factory_namespaces : frozenset of str
        Registered provider namespaces accepted for this operation.
    dtype_factory_kwargs : frozenset of str
        Factory keywords that use dtype resolution and normalization.
    runtime_arg_counts : frozenset of int
        Accepted positional counts before static controls are removed. The
        smallest count gives the number of ordinary operands.
    runtime_factory_kwargs : tuple of str
        Optional controls after the ordinary operands, in provider order.
        Controls supplied by keyword use this same order.
    runtime_factory_kw_prerequisites : tuple of tuple of str
        ``(control, required_control)`` pairs checked when a keyword
        control remains a runtime operand, such as padding needing a count.
        Factories remain responsible for validating static bindings.
    allowed_factory_kwargs : frozenset of str
        Keywords accepted for specialization, including scalar controls.
    required_factory_kwargs : frozenset of str
        Keywords that must be known after payload and launch inference.
    accepts_temp_storage : bool
        Whether argument splitting accepts a separate storage operand.
    scalar_binding_kwargs : frozenset of str
        Optional controls classified as static or runtime bindings. Other
        runtime controls are recorded as ``True`` when present.
    runtime_offset_kwarg : str or None
        Optional offset keyword handled after other runtime controls. A
        static offset becomes a binding; a runtime offset is last in the
        operand list. ``None`` disables this special case.
    infer_payload : callable
        Merge operand dtype and shape facts into factory inputs.
    analyze_match : callable or None
        Inspect a validated match and return family metadata for emission.
    prepare_runtime_args : callable or None
        Emit operand preparation, such as boxing a scalar Store value.
    validate_runtime_controls : callable or None
        Check control values and known dtypes before provider creation.

    Notes
    -----
    Construction copies collection fields and checks that keyword sets,
    positional counts, prerequisite names, and hook types agree. It does not
    validate an actual call or compile its provider.
    """

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
                    "runtime_factory_kw_prerequisite names "
                    "must be non-empty strings"
                )
            if not isinstance(required_name, str) or not required_name:
                raise ValueError(
                    "runtime_factory_kw_prerequisite names "
                    "must be non-empty strings"
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
                f"runtime_factory_kwargs must be allowed factory kwargs: "
                f"{names}"
            )
        base_runtime_arg_count = min(self.runtime_arg_counts)
        if max(self.runtime_arg_counts) - base_runtime_arg_count > len(
            self.runtime_factory_kwargs
        ):
            raise ValueError(
                "runtime_arg_counts require more trailing runtime arguments "
                "than runtime_factory_kwargs declares"
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
                    f"runtime_factory_kw_prerequisite targets must be "
                    f"runtime factory kwargs: {name}"
                )
            if required_name not in known_prerequisites:
                raise ValueError(
                    f"runtime_factory_kw_prerequisite requirements must be "
                    f"known factory kwargs: {required_name}"
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
                f"required_factory_kwargs must be allowed factory kwargs: "
                f"{names}"
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
                    "runtime_offset_kwarg must not also be "
                    "a runtime factory kwarg"
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
_REWRITE_OPERATIONS: dict[str, RewriteOperationSpecification] = {}
_GROUP_FAMILY_IMPORT_LOCK = RLock()


def group_operation(
    operation: str,
    *,
    family_module: str,
) -> Callable[[_CallableT], _CallableT]:
    """Associate a public group marker with its compiler family.

    This decorator records the exact callable object, so the planner
    recognizes aliases of the registered function without treating unrelated
    functions with the same name as cooperative operations. Record the family
    module for lazy loading of planning/rewrite hooks and attach the
    backend-member marker used during common API provenance checks.
    Registration does not import that family or wrap the decorated function.

    Parameters
    ----------
    operation : str
        Operation identifier shared by planning and rewrite lookup.
    family_module : str
        Compiler-family module that registers the operation's hooks.

    Returns
    -------
    callable
        Decorator that updates registries and callable metadata, then
        returns the original function. Equal registrations may repeat.

    Raises
    ------
    RuntimeError
        The callable already has another operation, or the operation
        already has another family module.
    """

    def decorate(function: _CallableT) -> _CallableT:
        existing = _GROUP_OPERATIONS.get(function)
        if existing is not None and existing != operation:
            raise RuntimeError(
                f"group marker {function!r} is already registered as "
                f"{existing!r}"
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
    """Import an operation's hooks only when a registry lookup needs them.

    Access the qualified backend member first if its marker has not yet
    registered a family. A reentrant lock serializes family imports.
    An unknown operation leaves no hooks.
    """

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
    results: tuple[GroupResultSource, ...] = (),
    validate_common_arguments: Callable[..., None] | None = None,
    result_resolver: Callable[[Any, Any], tuple[GroupResultSource, ...]]
    | None = None,
) -> None:
    """Register group-call lowering for one public operation.

    A primitive-family module calls this when it is imported. The group
    planner later looks up these hooks after resolving the call's thread
    group and binding its arguments. Registration connects an operation to
    the existing whole-function planner; it does not schedule another pass
    or compile a provider.

    Parameters
    ----------
    operation : str
        Public operation name used to find these hooks, such as ``"load"``.
    lower : callable
        Hook that receives the planning context, call assignment, resolved
        group, and bound arguments, and returns replacement IR statements.
    results : tuple of GroupResultSource, optional
        Result policies in return order, used to infer dtypes and per-thread
        element counts before replacement statements exist. One policy
        describes a direct result; several describe a tuple. An empty tuple
        supplies no result provenance.
    validate_common_arguments : callable or None
        Optional check for calls through the common ``cuda.coop`` API. It runs
        before ``lower`` with the planning context and bound arguments;
        backend-qualified calls skip it.
    result_resolver : callable or None
        Optional hook receiving the planning context and bound arguments and
        returning a replacement tuple of result policies. This lets a static
        selector choose the result layout; ``None`` uses ``results``.

    Raises
    ------
    TypeError
        A hook is not callable or a result entry is not a ``GroupResultSource``.
    RuntimeError
        Different hooks or result policies already use this operation name.
        Repeating the same registration is allowed.
    """

    registration = GroupPrimitiveRegistration(
        lower=lower,
        results=results,
        validate_common_arguments=validate_common_arguments,
        result_resolver=result_resolver,
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
    specification: RewriteOperationSpecification,
) -> None:
    """Register one provider ABI with the shared cooperative-call rewrite.

    Equal repeated specifications are allowed. Different specifications for
    one operation would make its calls ambiguous.

    Parameters
    ----------
    operation : str
        Operation name used to look up the call grammar and rewrite hooks.
    specification : RewriteOperationSpecification
        Provider call grammar and hooks for this operation.

    Raises
    ------
    TypeError
        ``specification`` is not a ``RewriteOperationSpecification``.
    RuntimeError
        The operation already has a different specification.
    """

    if not isinstance(specification, RewriteOperationSpecification):
        raise TypeError("specification must be a RewriteOperationSpecification")
    existing = _REWRITE_OPERATIONS.get(operation)
    if existing is not None and existing != specification:
        raise RuntimeError(
            f"rewrite operation {operation!r} is already registered"
        )
    _REWRITE_OPERATIONS[operation] = specification


def rewrite_operation(operation: str) -> RewriteOperationSpecification | None:
    """Return the cooperative-call rewrite registration for one operation.

    Load the owning primitive family when its registration is missing, without
    eagerly importing every family.

    Parameters
    ----------
    operation : str
        Operation name whose call grammar and rewrite hooks are requested.

    Returns
    -------
    RewriteOperationSpecification or None
        Registered specification, or ``None`` if no registration exists
        after loading the owning family.
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
    """Register a provider factory and its storage and thread scopes.

    The provider rewrite identifies factories by object identity and uses this
    metadata to validate provider calls and materialized invocables. The
    factory's import path or function name is not used to infer its ABI.
    Registration only declares metadata; the factory and source emitter must
    implement the declared storage and synchronization behavior.

    Parameters
    ----------
    function : callable
        Host-side callable that builds an ``Invocable`` from
        specialization keywords, or returns an ``Algorithm`` during batch
        collection. See the module overview for the distinction from
        public group markers.
    operation : str
        Non-empty operation identifier, such as ``"load"`` or ``"store"``.
    namespace : str
        Non-empty provider namespace, such as ``"block"`` or ``"warp"``.
    storage_abi : StorageABI
        Whether provider calls take a leading scratch pointer.
    execution_scope : SynchronizationScope
        Scope of threads executing the cooperative operation.
    synchronization_scope : SynchronizationScope
        Declared synchronization scope: ``NONE`` or the execution scope.

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
        Names, enum values, or the scope relationship are invalid.
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
            f"lowering factory {function!r} is already registered as "
            f"{existing!r}"
        )
    _FACTORY_OPERATIONS[function] = metadata
    return function


def factory_operation(function: Any) -> FactoryOperation | None:
    """Return metadata for an exactly registered lowering factory."""

    return _FACTORY_OPERATIONS.get(function)


__all__ = [
    "FactoryOperation",
    "GroupPrimitiveRegistration",
    "GroupResultSource",
    "RewriteOperationSpecification",
    "StorageABI",
    "expected_storage_reuse_barrier",
    "factory_operation",
    "group_operation",
    "group_operation_name",
    "group_primitive",
    "provider_synchronization_matches",
    "register_factory",
    "register_group_primitive",
    "register_rewrite_operation",
    "rewrite_operation",
]

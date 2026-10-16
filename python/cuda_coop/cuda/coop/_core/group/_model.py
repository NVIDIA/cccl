# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe a cooperative call and the requirements for implementing it.

A frontend first records an operation and its requested thread group in a
``GroupPrimitiveCall``. Launch resolution establishes the group's concrete
membership, then the operation's planner chooses an implementation and returns
a ``GroupLoweringPlan``. That plan carries the participation, synchronization,
and scratch-storage requirements needed to preserve the operation's semantics.
For example, a warp load using a transpose needs a separate scratch instance
for each participating warp and synchronization before that storage is reused.

Backends use these records to select a provider, compile specialized code, and
arrange storage and barriers. The records describe those requirements;
constructing them does not emit device code or check runtime participation.
Unsupported requests retain the original call and a structured reason so
callers can inspect a planning result before requiring executable support.

Implementation provenance identifies the native library entry point selected
by planning. Together with the specialized implementation and execution
requirements, it contributes to the plan's artifact identity. The logical
request and a particular implementation have separate keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .._algorithm import Algorithm
from .._symbols import semantic_token
from .._types import ParameterClassification
from ..launch import Dim3, LaunchFacts
from ..thread_group import MAPPED_GROUP_KINDS, ThreadGroup


class GroupLoweringTarget(str, Enum):
    """Implementation family chosen by planning, or an unsupported outcome.

    CUB block and warp targets describe the cooperative primitive's scope.
    A backend still selects and validates a provider for that target.
    """

    CUB_BLOCK = "cub_block"
    CUB_WARP = "cub_warp"
    UNSUPPORTED = "unsupported"


class GroupOperandKind(str, Enum):
    """Distinguish a scalar result from a per-member array result.

    The kind selects the result representation for lowering. A one-item array
    still has ``ARRAY`` kind; its extent alone does not make it a scalar.
    """

    SCALAR = "scalar"
    ARRAY = "array"


class ResultVisibility(str, Enum):
    """Which group members have meaningful results from the operation.

    ``ALL_MEMBERS`` describes a group result available to every member;
    ``GROUP_ROOT`` restricts it to the root; ``PER_MEMBER`` describes each
    member's own result, such as its items from a cooperative load.
    """

    ALL_MEMBERS = "all_members"
    GROUP_ROOT = "group_root"
    PER_MEMBER = "per_member"


class ResultOwnership(str, Enum):
    """Identify which members own the operation's logical result.

    ``EACH_MEMBER`` gives every member its own result. ``GROUP_ROOT`` assigns
    the result to rank zero only. Ownership describes the result contract; it
    does not allocate storage or imply that an input array is reused.
    """

    EACH_MEMBER = "each_member"
    GROUP_ROOT = "group_root"


class PreconditionEnforcement(str, Enum):
    """Whether planning checked a bound or the caller must satisfy it.

    ``PLANNER_VALIDATED`` records a check against a known static value.
    ``CALLER`` records a precondition on a runtime value; it does not request
    insertion of a runtime guard.
    """

    PLANNER_VALIDATED = "planner_validated"
    CALLER = "caller"


class StorageOwnership(str, Enum):
    """Who supplies the primitive's temporary storage.

    ``NONE`` selects a storage-free implementation. ``IMPLEMENTATION`` lets
    lowering arrange scratch internally; ``CALLER`` requires the supplied
    storage binding and its layout requests to be honored.
    """

    NONE = "none"
    IMPLEMENTATION = "implementation"
    CALLER = "caller"


class SynchronizationScope(str, Enum):
    """Members covered by execution or a storage-reuse synchronization.

    ``NONE`` requires no barrier, while ``WARP`` and ``BLOCK`` identify
    hardware scopes. ``GROUP`` refers to the resolved cooperative group;
    a backend must support that group's synchronization before lowering it.
    """

    NONE = "none"
    WARP = "warp"
    BLOCK = "block"
    GROUP = "group"


@dataclass(frozen=True)
class GroupTopologyRequirements:
    """Describe the group instances and ranks needed by backend lowering.

    Topology supplies the indexing rules for per-group scratch and data
    tiles. For logical warps of width 16 in a 128-thread block, for example,
    there are eight instances; dividing the linear thread rank by 16 selects
    the instance, and taking its remainder gives the rank within that group.
    The expression strings name canonical formulas understood by backends.
    They are not arbitrary expressions to evaluate as Python or C++.

    Attributes
    ----------
    group_kind : str
        Kind of the resolved group, such as ``"block"`` or ``"warp"``.
    logical_width : int
        Positive number of threads in each group instance.
    instances : int
        Positive number of group instances in the enclosing execution
        context; for block and warp lowering, this is per thread block.
    instance_index : str
        Symbolic rule selecting the current group instance, such as
        ``"cta"`` or ``"linear_thread_rank / 16"``.
    execution_scope : SynchronizationScope
        Scope within which the primitive's participating threads execute.
    thread_rank : str, optional
        Symbolic rule for the thread's rank within its group. Defaults to
        the block's linear thread rank.
    """

    group_kind: str
    logical_width: int
    instances: int
    instance_index: str
    execution_scope: SynchronizationScope
    thread_rank: str = field(default="linear_thread_rank", kw_only=True)

    def __post_init__(self) -> None:
        if not self.group_kind:
            raise ValueError("group topology kind must not be empty")
        for name in ("logical_width", "instances"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    f"group topology {name} must be a positive integer"
                )
        for name in ("instance_index", "thread_rank"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                label = name.replace("_", " ")
                raise ValueError(
                    f"group topology {label} must be a non-empty string"
                )
        object.__setattr__(
            self,
            "execution_scope",
            SynchronizationScope(self.execution_scope),
        )


class UnsupportedReasonCode(str, Enum):
    """Stable categories for unsupported planning and resolution outcomes.

    These distinguish missing launch facts, incomplete membership, and
    unsupported group or operation forms without matching diagnostic text.
    """

    MISSING_EXACT_BLOCK_DIM = "missing_exact_block_dim"
    PARTIAL_PHYSICAL_WARP = "partial_physical_warp"
    GROUP_KIND = "group_kind"
    OPERAND_FORM = "operand_form"
    OPERATION_VARIANT = "operation_variant"
    LAUNCH_CAPABILITY = "launch_capability"


class GroupOperationSemantics(Protocol):
    """Operation facts required by the common group-call model.

    A primitive-family record supplies its semantic identity, result
    visibility, and whether it returns a value. Its concrete type also has
    to be registered with the dispatcher, which provides that family's
    argument classification and planning functions. Satisfying this protocol
    alone does not register a new operation.
    """

    @property
    def semantic_key(self) -> tuple[Any, ...]: ...

    @property
    def result_visibility(self) -> ResultVisibility: ...

    @property
    def returns_value(self) -> bool: ...


def _requested_result_visibility(
    operation: GroupOperationSemantics,
) -> ResultVisibility:
    return ResultVisibility(operation.result_visibility)


def _group_key(group: ThreadGroup) -> tuple[Any, ...]:
    """Identify a group using the hierarchy dimensions relevant to its kind.

    Physical warps share the same logical identity across enclosing block
    shapes. Artifact identity separately retains the exact block dimensions
    needed for lowering and scratch layout.
    """

    hierarchy = group.hierarchy
    assert hierarchy is not None
    if group.kind == "warp":
        return "warp", "physical", 32
    if group.kind == "block":
        return "block", hierarchy.block_dim
    if group.kind == "cluster":
        return "cluster", hierarchy.block_dim, hierarchy.cluster_dim
    if group.kind == "grid":
        return (
            "grid",
            hierarchy.block_dim,
            hierarchy.cluster_dim,
            hierarchy.grid_dim,
        )
    if group.kind in MAPPED_GROUP_KINDS:
        return group.semantic_key
    return (group.kind,)


@dataclass(frozen=True, eq=False)
class GroupPrimitiveCall:
    """A requested group operation before selecting its implementation.

    Frontends provide a compile-time group description and a registered
    operation record. The dispatcher derives the argument classifications;
    the call itself contains no runtime operands or compiled provider.
    ``plan_group_primitive`` combines it with launch facts to resolve the
    group and select an implementation.

    Attributes
    ----------
    group : ThreadGroup
        Requested participating threads. Launch-dependent dimensions may
        still need to be resolved.
    operation : GroupOperationSemantics
        Primitive-family options, including static bindings and requested
        result behavior.
    argument_classifications : tuple of ParameterClassification
        Names, static/runtime kinds, and roles derived by the registered
        operation family. Computed during construction.

    Notes
    -----
    Equality and hashing use the requested group, the operation's semantic
    key, and its requested result visibility. Backend provider selection and
    executable-artifact details enter the later lowering plan.
    """

    group: ThreadGroup
    operation: GroupOperationSemantics
    argument_classifications: tuple[ParameterClassification, ...] = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("GroupPrimitiveCall group must be a ThreadGroup")
        from ._dispatch import _call_classifications, _is_group_operation

        if not _is_group_operation(self.operation):
            raise TypeError("unsupported GroupPrimitiveCall operation")
        object.__setattr__(
            self,
            "argument_classifications",
            _call_classifications(self.operation),
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            _group_key(self.group),
            self.operation.semantic_key,
            _requested_result_visibility(self.operation).value,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupPrimitiveCall):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class ArgumentPrecondition:
    """An inclusive scalar bound and who is responsible for checking it.

    Planners use these records for controls such as a tile's ``valid_items``
    count or a pointer offset. Construction checks the bounds' shape and
    ordering; it does not receive or validate the argument's actual value.

    Attributes
    ----------
    name : str
        Nonempty name of the argument constrained by this record.
    minimum, maximum : int or None
        Inclusive bounds. ``None`` leaves that end unconstrained here.
    enforcement : PreconditionEnforcement
        Whether planning already checked a static value or a runtime
        caller must satisfy the bound.
    """

    name: str
    minimum: int | None
    maximum: int | None
    enforcement: PreconditionEnforcement

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("argument precondition name must not be empty")
        object.__setattr__(
            self,
            "enforcement",
            PreconditionEnforcement(self.enforcement),
        )
        for bound_name, bound in (
            ("minimum", self.minimum),
            ("maximum", self.maximum),
        ):
            if bound is not None and (
                not isinstance(bound, int) or isinstance(bound, bool)
            ):
                raise TypeError(f"{bound_name} must be an integer or None")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("argument precondition minimum exceeds maximum")


@dataclass(frozen=True)
class ParticipationRequirements:
    """Membership and argument requirements for executing a group primitive.

    Backends use these facts to check that their provider and launch model
    match the selected group. Runtime requirements, such as converged entry
    and uniform arguments, remain obligations of the generated call; this
    record does not inspect running threads or insert validation code.

    Attributes
    ----------
    group_kind : str
        Kind of the resolved participating group.
    exact_group_size : int
        Number of participating threads in one group instance.
    exact_block_dim : Dim3 or None
        Exact enclosing launch shape when known, including all three axes.
    complete_membership : bool
        Whether the group has its full required membership.
    contiguous, aligned : bool
        Whether membership is contiguous in thread rank and aligned to
        the group's partition boundaries.
    converged_entry : bool
        Whether all participating threads must enter the primitive together.
    complete_parent_partition : bool
        Whether groups form complete partitions of the relevant parent.
    uniform_arguments : tuple of str, optional
        Argument names whose values must agree across participating members.
    valid_member_selection : str or None, optional
        Description of how a guarded operation selects valid data, such as
        ``"first valid_items tile elements"``. A partial data tile still
        requires the group's participating threads.
    argument_preconditions : tuple of ArgumentPrecondition, optional
        Static or caller-enforced scalar bounds. Names must be unique.
    """

    group_kind: str
    exact_group_size: int
    exact_block_dim: Dim3 | None
    complete_membership: bool
    contiguous: bool
    aligned: bool
    converged_entry: bool
    complete_parent_partition: bool
    uniform_arguments: tuple[str, ...] = ()
    valid_member_selection: str | None = None
    argument_preconditions: tuple[ArgumentPrecondition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "uniform_arguments", tuple(self.uniform_arguments)
        )
        object.__setattr__(
            self,
            "argument_preconditions",
            tuple(self.argument_preconditions),
        )
        if any(
            not isinstance(precondition, ArgumentPrecondition)
            for precondition in self.argument_preconditions
        ):
            raise TypeError(
                "argument_preconditions must contain "
                "ArgumentPrecondition records"
            )
        names = [
            precondition.name for precondition in self.argument_preconditions
        ]
        if len(names) != len(set(names)):
            raise ValueError("argument precondition names must be unique")


@dataclass(frozen=True, eq=False)
class LogicalResultContract:
    """Describe one named value returned by a cooperative operation.

    The backend uses this record to choose a scalar or array representation
    and to preserve who owns the result and where it is meaningful. It is
    metadata about a result, not the result value or an allocation request.
    For Exchange, for example, each member owns a new array with the input
    dtype and per-member item count.

    Attributes
    ----------
    name : str
        Non-empty role within the operation, such as ``"value"``.
    dtype : object
        Element dtype in the shared semantic model.
    visibility : ResultVisibility
        Members for which the result is meaningful. Per-member results may
        differ between members; an all-member result is a group result.
    ownership : ResultOwnership
        Whether each member owns a result or only the group root does.
    operand_kind : GroupOperandKind
        Scalar or array representation, independent of the number of items.
    items_per_member : int
        Positive element count. A scalar must contain exactly one item.
    root_rank : int or None, optional
        Required to be zero for a group-root result and absent otherwise. Root
        ownership and root visibility must agree.
    """

    name: str
    dtype: Any
    visibility: ResultVisibility
    ownership: ResultOwnership
    operand_kind: GroupOperandKind
    items_per_member: int
    root_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("logical result name must not be empty")
        object.__setattr__(
            self, "visibility", ResultVisibility(self.visibility)
        )
        object.__setattr__(self, "ownership", ResultOwnership(self.ownership))
        object.__setattr__(
            self, "operand_kind", GroupOperandKind(self.operand_kind)
        )
        if (
            not isinstance(self.items_per_member, int)
            or isinstance(self.items_per_member, bool)
            or self.items_per_member < 1
        ):
            raise ValueError("items_per_member must be a positive integer")
        if (
            self.operand_kind is GroupOperandKind.SCALAR
            and self.items_per_member != 1
        ):
            raise ValueError("scalar logical results contain exactly one item")
        is_root_result = self.ownership is ResultOwnership.GROUP_ROOT
        if is_root_result != (self.visibility is ResultVisibility.GROUP_ROOT):
            raise ValueError("group-root visibility and ownership must agree")
        if is_root_result:
            if (
                not isinstance(self.root_rank, int)
                or isinstance(self.root_rank, bool)
                or self.root_rank != 0
            ):
                raise ValueError("group-root results require root rank 0")
        elif self.root_rank is not None:
            raise ValueError("non-root results cannot define a root rank")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Return result identity with a normalized semantic dtype token."""

        return (
            self.name,
            semantic_token(self.dtype),
            self.visibility.value,
            self.ownership.value,
            self.operand_kind.value,
            self.items_per_member,
            self.root_rank,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogicalResultContract):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class ResultContract:
    """Group the named logical results in their public return order.

    ``values`` must be a non-empty sequence of ``LogicalResultContract``
    records with unique names. Its first entry is the primary result.
    ``primary``, ``visibility``, and ``operand_kind`` describe that entry;
    ``has_aggregate`` checks all entries. The contract does not prescribe
    how a backend packs multiple values.
    """

    values: tuple[LogicalResultContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError(
                "result contract requires at least one logical result"
            )
        if any(
            not isinstance(value, LogicalResultContract)
            for value in self.values
        ):
            raise TypeError("values must contain LogicalResultContract records")
        names = [value.name for value in self.values]
        if len(names) != len(set(names)):
            raise ValueError("logical result names must be unique")

    @property
    def primary(self) -> LogicalResultContract:
        return self.values[0]

    @property
    def visibility(self) -> ResultVisibility:
        return self.primary.visibility

    @property
    def operand_kind(self) -> GroupOperandKind:
        return self.primary.operand_kind

    @property
    def has_aggregate(self) -> bool:
        """Check for the named ``"aggregate"`` result."""

        return any(value.name == "aggregate" for value in self.values)


@dataclass(frozen=True)
class SynchronizationRequirements:
    """Entry and scratch-reuse synchronization required by the plan.

    Attributes
    ----------
    converged_entry : bool
        Whether participating threads must reach the operation together.
        Must agree with the plan's participation requirements.
    storage_reuse_barrier : SynchronizationScope
        Scope of synchronization required before scratch is reused, or
        ``NONE`` when no automatic reuse barrier is requested. This does not
        describe every synchronization performed inside the native primitive.
    """

    converged_entry: bool
    storage_reuse_barrier: SynchronizationScope


@dataclass(frozen=True)
class TempStorageRequirements:
    """Describe scratch ownership, per-group layout, and reuse policy.

    A storage-bearing primitive may need one independent scratch instance
    per participating group. Planning records that topology and any caller
    requests; backend materialization discovers the concrete layout and
    lowering arranges the storage binding. Constructing this record does not
    allocate memory or establish that a requested capacity is sufficient.

    Attributes
    ----------
    ownership : StorageOwnership
        Storage-free, internally arranged, or caller-supplied storage.
    address_space : str or None
        Required memory space, such as ``"shared"`` for CUB scratch.
    cpp_type : str or None
        Native scratch type when already named. ``None`` can defer its
        resolution until the specialized implementation is materialized.
    instances : int or None
        Positive scratch-instance count for a storage-bearing operation.
    instance_index : str or None
        Symbolic rule selecting the current group's scratch instance.
    exact_layout_required : bool
        Whether the supplied storage must be checked against the concrete
        implementation layout, as required for caller-owned storage.
    sharing : {"shared", "exclusive"} or None, optional
        Caller storage-sharing policy: reusable with other compatible
        consumers or reserved exclusively. ``None`` for implementation-owned
        storage; distinct from the ``"shared"`` address space.
    requested_size_in_bytes : int or None, optional
        Positive caller-requested capacity, or no explicit capacity request.
    requested_alignment : int or None, optional
        Positive caller-requested byte alignment, or no explicit request.
    auto_sync : bool, optional
        Whether lowering should arrange scratch-reuse synchronization.
        Defaults to ``True``; storage-free operations must set it to ``False``.

    Notes
    -----
    Storage-free requirements carry no layout, sharing, or size requests.
    Implementation-owned storage carries no caller sharing, size, or
    alignment requests. Caller-owned storage must select a sharing policy.
    """

    ownership: StorageOwnership
    address_space: str | None
    cpp_type: str | None
    instances: int | None
    instance_index: str | None
    exact_layout_required: bool
    sharing: str | None = None
    requested_size_in_bytes: int | None = None
    requested_alignment: int | None = None
    auto_sync: bool = True

    def __post_init__(self) -> None:
        """Check that layout and reuse requests agree with storage ownership.

        A storage-free operation cannot carry layout requests. An internally
        managed allocation needs group-instance information, while
        caller-owned storage also needs a sharing policy. Concrete capacity is
        checked later.
        """

        object.__setattr__(self, "ownership", StorageOwnership(self.ownership))
        if self.sharing not in {None, "shared", "exclusive"}:
            raise ValueError(
                "temporary storage sharing must be shared or exclusive"
            )
        if not isinstance(self.auto_sync, bool):
            raise TypeError("auto_sync must be a bool")
        if self.ownership is StorageOwnership.NONE:
            if any(
                value is not None
                for value in (
                    self.address_space,
                    self.cpp_type,
                    self.instances,
                    self.instance_index,
                    self.sharing,
                    self.requested_size_in_bytes,
                    self.requested_alignment,
                )
            ):
                raise ValueError(
                    "storage-free requirements cannot carry storage layout"
                )
            if self.exact_layout_required:
                raise ValueError(
                    "storage-free requirements cannot require an exact layout"
                )
            if self.auto_sync:
                raise ValueError(
                    "storage-free requirements cannot request automatic sync"
                )
        else:
            if (
                not isinstance(self.instances, int)
                or isinstance(self.instances, bool)
                or self.instances < 1
            ):
                raise ValueError(
                    "storage-bearing contracts require "
                    "a positive instance count"
                )
            if (
                not isinstance(self.instance_index, str)
                or not self.instance_index
            ):
                raise ValueError(
                    "storage-bearing contracts require "
                    "a non-empty instance index"
                )
        if self.ownership is StorageOwnership.IMPLEMENTATION:
            if self.sharing is not None:
                raise ValueError(
                    "implementation-owned storage has no sharing mode"
                )
            if self.requested_size_in_bytes is not None:
                raise ValueError(
                    "implementation-owned storage has no requested size"
                )
            if self.requested_alignment is not None:
                raise ValueError(
                    "implementation-owned storage has no requested alignment"
                )
        elif self.ownership is StorageOwnership.CALLER and self.sharing is None:
            raise ValueError("caller-owned storage requires a sharing mode")
        for name in ("requested_size_in_bytes", "requested_alignment"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True)
class GroupExecutionRequirements:
    """Group topology, participation, synchronization, and scratch requirements.

    Operation planners attach these records to the selected implementation's
    lowering plan so backends can arrange execution and temporary storage.
    """

    topology: GroupTopologyRequirements
    participation: ParticipationRequirements
    synchronization: SynchronizationRequirements
    temp_storage: TempStorageRequirements


@dataclass(frozen=True)
class ImplementationProvenance:
    """Identify the native library entry point chosen for a group operation.

    A logical request such as a group load does not by itself identify the
    native primitive that will execute it. The planner records that choice
    here so a backend can recognize the implementation and route it to a
    supported provider. For example, a CUB block load records ``"CUB"``,
    ``"cub/block/block_load.cuh"``, ``"cub::BlockLoad"``, and ``"Load"``.
    A backend can use this tuple to select its load or store route, then check
    the provider against the plan's requirements.

    The same native entry point can serve many element types, tile sizes,
    and algorithm choices. Those details belong to the plan's specialized
    ``implementation`` and its metadata. Provenance supplies the entry-point
    identity; both contribute to ``GroupLoweringPlan.artifact_key`` so the
    identity retains the chosen native implementation as well as its
    specialization and execution requirements.

    Attributes
    ----------
    library : str
        Underlying native library, such as ``"CUB"``. The backend compiler
        or Python provider factory is selected separately.
    header : str
        Native header declaring the chosen primitive.
    cpp_class : str
        Qualified C++ class or namespace containing the operation, such as
        ``"cub::BlockLoad"``.
    method : str
        Native operation name invoked by the implementation, such as
        ``"Load"``.

    Notes
    -----
    This record identifies an implementation choice. It does not record a
    library version, compiled binary, or proof that a backend supports the
    choice. A backend can reject an unrecognized tuple; planning failures
    are represented separately by ``UnsupportedReason``.
    """

    library: str
    header: str
    cpp_class: str
    method: str

    @property
    def semantic_key(self) -> tuple[str, str, str, str]:
        """Return the library/header/implementation/method identity."""

        return self.library, self.header, self.cpp_class, self.method


@dataclass(frozen=True)
class UnsupportedReason:
    """A machine-readable failure category with a human explanation.

    ``code`` identifies the unsupported case; ``message`` supplies details
    for diagnostics and ``require_supported`` exceptions. Messages are
    excluded from equality and hashing, allowing wording to change without
    changing the failure's identity.
    """

    code: UnsupportedReasonCode
    message: str = field(compare=False, hash=False)


@dataclass(frozen=True)
class ThreadGroupLaunchResolution:
    """A thread group reconciled with launch facts, or a resolution failure.

    Resolution establishes dimensions and membership needed by subsequent
    operation planning. It does not choose a primitive implementation.

    Attributes
    ----------
    group : ThreadGroup
        Resolved group on success, or the group retained with failure context.
    unsupported : UnsupportedReason or None, optional
        Why the launch facts cannot support this group, or ``None`` when
        resolution succeeded.
    """

    group: ThreadGroup
    unsupported: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError(
                "ThreadGroupLaunchResolution group must be a ThreadGroup"
            )
        if self.unsupported is not None and not isinstance(
            self.unsupported, UnsupportedReason
        ):
            raise TypeError(
                "ThreadGroupLaunchResolution unsupported must be an "
                "UnsupportedReason"
            )

    def require_supported(self) -> ThreadGroup:
        """Return the resolved group or raise its ``NotImplementedError``."""

        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self.group


@dataclass(frozen=True, eq=False)
class GroupLoweringPlan:
    """An implementation choice and its execution requirements.

    Operation planners construct this after resolving the requested group
    against launch facts. A supported plan combines a specialized native
    implementation with topology, participation, synchronization, storage,
    and provenance. Backends use that information to select compatible
    providers and produce code; a supported planning outcome alone does not
    mean that compilation has occurred or every backend can lower the plan.

    Construction checks consistency between the resolved group and its
    topology, participation, and convergence requirements. Unsupported plans
    instead retain a typed reason and can be inspected before
    ``require_supported`` turns that reason into an exception.

    Attributes
    ----------
    target : GroupLoweringTarget
        Chosen implementation family, or ``UNSUPPORTED``.
    call : GroupPrimitiveCall
        Original request, retained alongside the resolved group.
    resolved_group : ThreadGroup
        Group after applying launch facts, or the group retained on failure.
    implementation : Algorithm or None
        Specialized primitive description; ``None`` for an unsupported plan.
    topology : GroupTopologyRequirements or None
        Group instances and rank rules used for indexing and execution.
    participation : ParticipationRequirements or None
        Required membership, launch shape, uniformity, and scalar bounds.
    result : ResultContract or None
        Returned-value descriptions for supported operations that return a
        value. Load/Store writes through its destination and uses ``None``.
    synchronization : SynchronizationRequirements or None
        Converged-entry and scratch-reuse requirements.
    temp_storage : TempStorageRequirements or None
        Scratch ownership, layout requests, and automatic reuse policy.
    provenance : ImplementationProvenance or None
        Native library entry point used for routing and artifact identity.
    unsupported : UnsupportedReason or None, optional
        Required for ``UNSUPPORTED`` and absent for a supported plan.

    Notes
    -----
    Supported plans specify all lowering requirements. Their equality and
    hashing use ``artifact_key``. Unsupported plans have no artifact key;
    they compare by logical semantics and reason code, excluding diagnostic
    wording. ``semantic_key`` alone is not enough to identify an executable
    implementation because it omits provider and storage choices.
    """

    target: GroupLoweringTarget
    call: GroupPrimitiveCall
    resolved_group: ThreadGroup
    implementation: Algorithm | None
    topology: GroupTopologyRequirements | None
    participation: ParticipationRequirements | None
    result: ResultContract | None
    synchronization: SynchronizationRequirements | None
    temp_storage: TempStorageRequirements | None
    provenance: ImplementationProvenance | None
    unsupported: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        """Require a complete implementation plan or an unsupported reason.

        For a supported plan, compare the execution requirements with the
        resolved group. This catches inconsistent group sizes, scratch
        indexing, launch dimensions, and convergence requirements before a
        backend consumes them.
        """

        is_unsupported = self.target is GroupLoweringTarget.UNSUPPORTED
        if is_unsupported != (self.unsupported is not None):
            raise ValueError("unsupported plans require exactly one reason")
        result_required = self.call.operation.returns_value
        if not isinstance(result_required, bool):
            raise TypeError("operation returns_value must be a bool")
        if not is_unsupported and (
            self.implementation is None
            or self.topology is None
            or self.participation is None
            or self.synchronization is None
            or self.temp_storage is None
            or self.provenance is None
            or (result_required and self.result is None)
        ):
            raise ValueError(
                "supported plans require complete lowering requirements"
            )
        if not is_unsupported:
            assert self.topology is not None
            assert self.participation is not None
            resolved_kind = self.resolved_group.kind
            if (
                self.topology.group_kind != resolved_kind
                or self.participation.group_kind != resolved_kind
            ):
                raise ValueError(
                    "supported plan group requirements must match "
                    "the resolved group kind"
                )
            resolved_size = self.resolved_group.static_size
            if resolved_size is None or (
                self.topology.logical_width != resolved_size
                or self.participation.exact_group_size != resolved_size
            ):
                raise ValueError(
                    "supported plan group widths must match "
                    "the resolved group size"
                )
            resolved_block_dim = self.resolved_group.block_dim
            if (
                resolved_block_dim is not None
                and self.participation.exact_block_dim != resolved_block_dim
            ):
                raise ValueError(
                    "supported plan block dimensions must match "
                    "the resolved group"
                )
            from ._execution_requirements import _group_topology

            expected_topology = _group_topology(
                self.resolved_group,
                LaunchFacts(exact_block_dim=self.participation.exact_block_dim),
            )
            if self.topology != expected_topology:
                raise ValueError(
                    "supported plan topology must match the resolved group"
                )
            complete_membership = (
                self.resolved_group.complete_membership is not False
            )
            complete_parent_partition = (
                resolved_kind == "warp"
                or self.resolved_group.complete_membership is True
            )
            if (
                self.participation.complete_membership
                is not complete_membership
                or self.participation.complete_parent_partition
                is not complete_parent_partition
            ):
                raise ValueError(
                    "supported plan participation must match the resolved group"
                )
            assert self.synchronization is not None
            if (
                self.participation.converged_entry
                is not self.synchronization.converged_entry
            ):
                raise ValueError(
                    "supported plan participation and synchronization "
                    "must agree on converged entry"
                )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify the resolved request without its provider choice.

        Physical warp requests can share this key across enclosing block
        shapes. Use ``artifact_key`` when execution and storage details must
        also distinguish plans.
        """

        result_visibility = (
            None if self.result is None else self.result.visibility.value
        )
        return (
            _group_key(self.resolved_group),
            self.call.operation.semantic_key,
            result_visibility,
        )

    @property
    def artifact_key(self) -> tuple[Any, ...] | None:
        """Identify a supported implementation and its lowering requirements.

        Include exact block dimensions, the specialized implementation,
        execution/storage requirements, and native provenance. Backends can use
        this identity when reusing generated artifacts, together with their
        compiler- and target-specific cache inputs. Unsupported plans return
        ``None`` because they have no executable implementation to reuse.
        """

        if self.unsupported is not None:
            return None
        implementation_key = (
            None
            if self.implementation is None
            else self.implementation.semantic_key
        )
        return (
            self.target.value,
            _group_key(self.resolved_group),
            self.resolved_group.hierarchy.block_dim,
            self.topology,
            self.call.operation.semantic_key,
            implementation_key,
            self.participation,
            self.result,
            self.synchronization,
            self.temp_storage,
            None if self.provenance is None else self.provenance.semantic_key,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupLoweringPlan):
            return NotImplemented
        return self._identity_key == other._identity_key

    def __hash__(self) -> int:
        return hash(self._identity_key)

    @property
    def _identity_key(self) -> tuple[Any, ...]:
        # Diagnostic prose can evolve without changing an unsupported request;
        # successful requests must retain every artifact-affecting requirement.
        if self.artifact_key is not None:
            return "artifact", self.artifact_key
        assert self.unsupported is not None
        return "unsupported", self.semantic_key, self.unsupported.code.value

    def require_supported(self) -> GroupLoweringPlan:
        """Return a supported plan or report its reason in an exception.

        Unsupported plans raise ``NotImplementedError`` with their message.
        """

        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self


__all__ = [
    "ArgumentPrecondition",
    "GroupExecutionRequirements",
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupOperandKind",
    "GroupOperationSemantics",
    "GroupPrimitiveCall",
    "GroupTopologyRequirements",
    "ImplementationProvenance",
    "LogicalResultContract",
    "ParticipationRequirements",
    "PreconditionEnforcement",
    "ResultContract",
    "ResultOwnership",
    "ResultVisibility",
    "StorageOwnership",
    "SynchronizationRequirements",
    "SynchronizationScope",
    "TempStorageRequirements",
    "ThreadGroupLaunchResolution",
    "UnsupportedReason",
    "UnsupportedReasonCode",
]

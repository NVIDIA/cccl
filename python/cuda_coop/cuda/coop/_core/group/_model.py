# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared identity and contract records for thread-group lowering.

Primitive semantics and planner choices live in the adjacent family modules.
This module owns only the cross-family records whose exact key shapes are
consumed by backend caches and artifact generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .._algorithm import AlgorithmSpec
from .._symbols import semantic_token
from .._types import ParameterClassification
from ..launch import Dim3
from ..thread_group import MAPPED_GROUP_KINDS, ThreadGroup

if TYPE_CHECKING:
    from ._dispatch import GroupOperationSemantics


class GroupLoweringTarget(str, Enum):
    CUDAX_GROUP = "cudax_group"
    CUB_BLOCK = "cub_block"
    CUB_WARP = "cub_warp"
    UNSUPPORTED = "unsupported"


class GroupOperandKind(str, Enum):
    SCALAR = "scalar"
    ARRAY = "array"


class ResultVisibility(str, Enum):
    ALL_MEMBERS = "all_members"
    GROUP_ROOT = "group_root"
    PER_MEMBER = "per_member"


class ResultOwnership(str, Enum):
    EACH_MEMBER = "each_member"
    GROUP_ROOT = "group_root"


class PreconditionEnforcement(str, Enum):
    PLANNER_VALIDATED = "planner_validated"
    CALLER = "caller"


class StorageOwnership(str, Enum):
    IMPLEMENTATION = "implementation"
    CALLER = "caller"


class SynchronizationScope(str, Enum):
    NONE = "none"
    WARP = "warp"
    BLOCK = "block"
    GROUP = "group"


class UnsupportedReasonCode(str, Enum):
    MISSING_EXACT_BLOCK_DIM = "missing_exact_block_dim"
    PARTIAL_PHYSICAL_WARP = "partial_physical_warp"
    GROUP_KIND = "group_kind"
    OPERAND_FORM = "operand_form"
    CUB_BROADCAST = "cub_broadcast"
    OPERATION_VARIANT = "operation_variant"
    LAUNCH_CAPABILITY = "launch_capability"


class CudaxReturnKind(str, Enum):
    VALUE = "value"
    OPTIONAL_VALUE = "optional_value"


def _lowered_operation_semantic_key(
    operation: GroupOperationSemantics,
    target: GroupLoweringTarget,
) -> tuple[Any, ...]:
    from ..block.scan import BlockScanAlgorithm
    from .scan import GroupScanSemantics

    if (
        isinstance(operation, GroupScanSemantics)
        and target is GroupLoweringTarget.CUB_BLOCK
    ):
        algorithm = operation.cub_algorithm or BlockScanAlgorithm.RAKING
        return operation.primitive.semantic_key, algorithm.value
    return operation.semantic_key


def _requested_result_visibility(
    operation: GroupOperationSemantics,
) -> ResultVisibility:
    from .reduce import GroupReduceSemantics

    if isinstance(operation, GroupReduceSemantics):
        return (
            ResultVisibility.ALL_MEMBERS
            if operation.broadcast
            else ResultVisibility.GROUP_ROOT
        )
    return ResultVisibility.PER_MEMBER


def _group_key(group: ThreadGroup) -> tuple[Any, ...]:
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
    group: ThreadGroup
    operation: GroupOperationSemantics
    argument_classifications: tuple[ParameterClassification, ...] = field(init=False)

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
class CudaxCallDescription:
    primitive: str
    header: str
    namespace: str
    overload: str | None = None
    parameters: tuple[ParameterClassification, ...] = ()
    return_kind: CudaxReturnKind = CudaxReturnKind.VALUE

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "return_kind", CudaxReturnKind(self.return_kind))
        if any(
            not isinstance(parameter, ParameterClassification)
            for parameter in self.parameters
        ):
            raise TypeError("CUDAX parameters must be ParameterClassification records")
        forbidden = {"group", "launch", "launch_facts"}
        if any(parameter.name in forbidden for parameter in self.parameters):
            raise ValueError("CUDAX runtime ABI cannot contain group or launch markers")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive,
            self.header,
            self.namespace,
            self.overload,
            self.parameters,
            self.return_kind.value,
        )


@dataclass(frozen=True)
class ArgumentPrecondition:
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

    def validate(self, value: int) -> None:
        """Validate a concrete value when a caller can inspect it."""

        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{self.name} must be an integer")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.name} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.name} must be at most {self.maximum}")


@dataclass(frozen=True)
class ParticipationContract:
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
        object.__setattr__(self, "uniform_arguments", tuple(self.uniform_arguments))
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
                "argument_preconditions must contain ArgumentPrecondition records"
            )
        names = [precondition.name for precondition in self.argument_preconditions]
        if len(names) != len(set(names)):
            raise ValueError("argument precondition names must be unique")


@dataclass(frozen=True, eq=False)
class LogicalResultContract:
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
        object.__setattr__(self, "visibility", ResultVisibility(self.visibility))
        object.__setattr__(self, "ownership", ResultOwnership(self.ownership))
        object.__setattr__(self, "operand_kind", GroupOperandKind(self.operand_kind))
        if (
            not isinstance(self.items_per_member, int)
            or isinstance(self.items_per_member, bool)
            or self.items_per_member < 1
        ):
            raise ValueError("items_per_member must be a positive integer")
        if self.operand_kind is GroupOperandKind.SCALAR and self.items_per_member != 1:
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
    values: tuple[LogicalResultContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("result contract requires at least one logical result")
        if any(not isinstance(value, LogicalResultContract) for value in self.values):
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
    def result_items_per_thread(self) -> int:
        return self.primary.items_per_member

    @property
    def has_aggregate(self) -> bool:
        return any(value.name == "aggregate" for value in self.values)


@dataclass(frozen=True)
class SynchronizationContract:
    converged_entry: bool
    storage_reuse_barrier: SynchronizationScope


@dataclass(frozen=True)
class TempStorageContract:
    ownership: StorageOwnership
    address_space: str | None
    cpp_type: str | None
    instances: int | None
    instance_index: str | None
    exact_layout_required: bool


@dataclass(frozen=True)
class ImplementationProvenance:
    library: str
    header: str
    cpp_class: str
    method: str

    @property
    def semantic_key(self) -> tuple[str, str, str, str]:
        return self.library, self.header, self.cpp_class, self.method


@dataclass(frozen=True)
class UnsupportedReason:
    code: UnsupportedReasonCode
    message: str = field(compare=False, hash=False)


@dataclass(frozen=True)
class ThreadGroupResolution:
    """One launch-reconciled static group or a typed unsupported reason."""

    group: ThreadGroup
    unsupported: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("ThreadGroupResolution group must be a ThreadGroup")
        if self.unsupported is not None and not isinstance(
            self.unsupported, UnsupportedReason
        ):
            raise TypeError(
                "ThreadGroupResolution unsupported must be an UnsupportedReason"
            )

    def require_supported(self) -> ThreadGroup:
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self.group


@dataclass(frozen=True, eq=False)
class GroupLoweringPlan:
    target: GroupLoweringTarget
    call: GroupPrimitiveCall
    resolved_group: ThreadGroup
    implementation: CudaxCallDescription | AlgorithmSpec | None
    participation: ParticipationContract | None
    result: ResultContract | None
    synchronization: SynchronizationContract | None
    temp_storage: TempStorageContract | None
    provenance: ImplementationProvenance | None
    unsupported: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        is_unsupported = self.target is GroupLoweringTarget.UNSUPPORTED
        if is_unsupported != (self.unsupported is not None):
            raise ValueError("unsupported plans require exactly one reason")
        from .load_store import GroupLoadStoreKind, GroupLoadStoreSemantics

        result_required = not (
            isinstance(self.call.operation, GroupLoadStoreSemantics)
            and self.call.operation.kind is GroupLoadStoreKind.STORE
        )
        if not is_unsupported and (
            self.implementation is None
            or self.participation is None
            or self.synchronization is None
            or self.temp_storage is None
            or self.provenance is None
            or (result_required and self.result is None)
        ):
            raise ValueError("supported plans require complete lowering contracts")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        result_visibility = (
            None if self.result is None else self.result.visibility.value
        )
        return (
            _group_key(self.resolved_group),
            _lowered_operation_semantic_key(self.call.operation, self.target),
            result_visibility,
        )

    @property
    def artifact_key(self) -> tuple[Any, ...] | None:
        if self.unsupported is not None:
            return None
        implementation_key = (
            None
            if self.implementation is None
            else getattr(self.implementation, "semantic_key")
        )
        return (
            self.target.value,
            _group_key(self.resolved_group),
            self.resolved_group.hierarchy.block_dim,
            _lowered_operation_semantic_key(self.call.operation, self.target),
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
        if self.artifact_key is not None:
            return "artifact", self.artifact_key
        assert self.unsupported is not None
        return "unsupported", self.semantic_key, self.unsupported.code.value

    def require_supported(self) -> "GroupLoweringPlan":
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self


__all__ = [
    "ArgumentPrecondition",
    "CudaxCallDescription",
    "CudaxReturnKind",
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupOperandKind",
    "GroupPrimitiveCall",
    "ImplementationProvenance",
    "LogicalResultContract",
    "ParticipationContract",
    "PreconditionEnforcement",
    "ResultContract",
    "ResultOwnership",
    "ResultVisibility",
    "StorageOwnership",
    "SynchronizationContract",
    "SynchronizationScope",
    "TempStorageContract",
    "ThreadGroupResolution",
    "UnsupportedReason",
    "UnsupportedReasonCode",
]

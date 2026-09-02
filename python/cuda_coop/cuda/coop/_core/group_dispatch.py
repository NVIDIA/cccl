# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed planning for block-wide scalar reduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any

from ._bindings import ArgumentBinding, BindingKind
from ._symbols import semantic_token
from .block.reduce import (
    BlockReduceAlgorithm,
    BlockReduceOperation,
    BlockReduceOperator,
    BlockReduceSpec,
    make_block_reduce_spec,
    normalize_block_reduce_algorithm,
    normalize_block_reduce_operator,
)
from .launch import Dim3, LaunchFacts
from .thread_group import ThreadGroup, ThreadHierarchy

GroupReduceAlgorithm = BlockReduceAlgorithm
GroupReduceOperation = BlockReduceOperation
GroupReduceOperator = BlockReduceOperator


class GroupLoweringTarget(str, Enum):
    """Backend provider selected for one group primitive."""

    CUB_BLOCK = "cub_block"
    UNSUPPORTED = "unsupported"


class ResultVisibility(str, Enum):
    """Members for which a collective result is defined."""

    ROOT_ONLY = "root_only"


class ResultOwnership(str, Enum):
    """Member that owns the collective result."""

    GROUP_ROOT = "group_root"


class GroupOperandKind(str, Enum):
    """Portable value representation consumed per thread."""

    SCALAR = "scalar"


class StorageOwnership(str, Enum):
    """Owner of temporary storage for one provider call."""

    IMPLEMENTATION = "implementation"


class SynchronizationScope(str, Enum):
    """Synchronization required before temporary-storage reuse."""

    BLOCK = "block"


class UnsupportedReasonCode(str, Enum):
    """Stable reason codes for fail-closed planning."""

    MISSING_EXACT_BLOCK_DIM = "missing_exact_block_dim"
    UNVERIFIED_EXACT_BLOCK_DIM = "unverified_exact_block_dim"


@dataclass(frozen=True, eq=False)
class GroupReduceSemantics:
    """Backend-neutral semantics of one block-wide scalar reduction."""

    dtype: Any
    operation: GroupReduceOperation = GroupReduceOperation.REDUCE
    binary_op: GroupReduceOperator = GroupReduceOperator.SUM
    algorithm: GroupReduceAlgorithm = GroupReduceAlgorithm.WARP_REDUCTIONS
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", GroupReduceOperation(self.operation))
        object.__setattr__(
            self,
            "binary_op",
            normalize_block_reduce_operator(self.binary_op),
        )
        object.__setattr__(
            self,
            "algorithm",
            normalize_block_reduce_algorithm(self.algorithm),
        )
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
        if self.operation is GroupReduceOperation.SUM and (
            self.binary_op is not GroupReduceOperator.SUM
        ):
            raise ValueError("cuda.coop.sum requires the sum operator")

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            f"group_{self.operation.value}",
            semantic_token(self.dtype),
            self.binary_op.value,
            self.algorithm.value,
            self.valid_items,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupReduceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupPrimitiveCall:
    """One canonical group reduction call before launch resolution."""

    group: ThreadGroup
    operation: GroupReduceSemantics
    source: str = field(default="canonical", compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("GroupPrimitiveCall group must be a ThreadGroup")
        if not isinstance(self.operation, GroupReduceSemantics):
            raise TypeError("GroupPrimitiveCall operation must be a reduction")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.group.semantic_key, self.operation.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupPrimitiveCall):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class ParticipationContract:
    """Participation requirements for one block collective."""

    group_kind: str
    exact_group_size: int
    exact_block_dim: Dim3
    complete_membership: bool
    converged_entry: bool
    uniform_arguments: tuple[str, ...] = ()
    valid_member_selection: str | None = None


@dataclass(frozen=True, eq=False)
class ResultContract:
    """Root-only scalar result contract."""

    dtype: Any
    visibility: ResultVisibility = ResultVisibility.ROOT_ONLY
    ownership: ResultOwnership = ResultOwnership.GROUP_ROOT
    operand_kind: GroupOperandKind = GroupOperandKind.SCALAR
    root_rank: int = 0

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            semantic_token(self.dtype),
            self.visibility.value,
            self.ownership.value,
            self.operand_kind.value,
            self.root_rank,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultContract):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class SynchronizationContract:
    """Synchronization requirements exposed by the provider plan."""

    converged_entry: bool
    storage_reuse_barrier: SynchronizationScope


@dataclass(frozen=True)
class TempStorageContract:
    """Temporary-storage ownership for the provider call."""

    ownership: StorageOwnership


@dataclass(frozen=True)
class ImplementationProvenance:
    """Upstream implementation selected by the plan."""

    library: str
    header: str
    cpp_class: str
    method: str

    @property
    def semantic_key(self) -> tuple[str, str, str, str]:
        return self.library, self.header, self.cpp_class, self.method


@dataclass(frozen=True)
class UnsupportedReason:
    """Typed reason that a call cannot be planned safely."""

    code: UnsupportedReasonCode
    message: str = field(compare=False, hash=False)


@dataclass(frozen=True)
class ThreadGroupResolution:
    """Resolved block descriptor or a typed unsupported reason."""

    group: ThreadGroup
    unsupported: UnsupportedReason | None = None

    def require_supported(self) -> ThreadGroup:
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self.group


@dataclass(frozen=True, eq=False)
class GroupLoweringPlan:
    """Complete backend-neutral lowering contract for one reduction."""

    target: GroupLoweringTarget
    call: GroupPrimitiveCall
    resolved_group: ThreadGroup
    implementation: BlockReduceSpec | None
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
        if not is_unsupported and any(
            item is None
            for item in (
                self.implementation,
                self.participation,
                self.result,
                self.synchronization,
                self.temp_storage,
                self.provenance,
            )
        ):
            raise ValueError("supported plans require complete lowering contracts")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.resolved_group.semantic_key, self.call.operation.semantic_key

    @property
    def artifact_key(self) -> tuple[Any, ...]:
        if self.unsupported is not None:
            return self.target.value, "unsupported", self.unsupported.code.value
        assert self.implementation is not None
        assert self.participation is not None
        assert self.result is not None
        assert self.synchronization is not None
        assert self.temp_storage is not None
        assert self.provenance is not None
        return (
            self.target.value,
            self.resolved_group.semantic_key,
            self.implementation.semantic_key,
            self.call.operation.semantic_key,
            self.participation,
            self.result,
            self.synchronization,
            self.temp_storage,
            self.provenance.semantic_key,
        )

    def require_supported(self) -> GroupLoweringPlan:
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupLoweringPlan):
            return NotImplemented
        return (
            self.artifact_key == other.artifact_key
            and self.semantic_key == other.semantic_key
        )

    def __hash__(self) -> int:
        return hash((self.artifact_key, self.semantic_key))


def make_group_primitive_call(
    group: ThreadGroup,
    operation: GroupReduceSemantics,
    *,
    source: str = "canonical",
) -> GroupPrimitiveCall:
    """Build one canonical group reduction call."""

    return GroupPrimitiveCall(group=group, operation=operation, source=source)


def resolve_thread_group(
    group: ThreadGroup,
    launch: LaunchFacts,
) -> ThreadGroupResolution:
    """Resolve the current block against exact compiler launch facts."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("group must be a ThreadGroup")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")
    if launch.exact_block_dim is None:
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM,
                "block reduction requires exact block dimensions",
            ),
        )
    if not launch.is_verified("exact_block_dim"):
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.UNVERIFIED_EXACT_BLOCK_DIM,
                "block reduction requires compiler-verified exact block dimensions",
            ),
        )
    existing = group.hierarchy.block_dim
    if existing is not None and existing != launch.exact_block_dim:
        raise ValueError(
            f"group block dimensions {existing!r} do not match exact launch "
            f"dimensions {launch.exact_block_dim!r}"
        )
    resolved = group.with_hierarchy(
        ThreadHierarchy(block_dim=launch.exact_block_dim),
        source="launch_facts",
    )
    return ThreadGroupResolution(resolved)


def _unsupported_plan(
    call: GroupPrimitiveCall,
    resolution: ThreadGroupResolution,
) -> GroupLoweringPlan:
    assert resolution.unsupported is not None
    return GroupLoweringPlan(
        target=GroupLoweringTarget.UNSUPPORTED,
        call=call,
        resolved_group=resolution.group,
        implementation=None,
        participation=None,
        result=None,
        synchronization=None,
        temp_storage=None,
        provenance=None,
        unsupported=resolution.unsupported,
    )


def _validate_static_valid_items(
    binding: ArgumentBinding,
    *,
    group_size: int | None = None,
) -> None:
    if binding.kind is not BindingKind.STATIC:
        return
    value = binding.value
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"valid_items must be an integer, not {type(value).__name__}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError("valid_items must be at least 1")
    if group_size is not None and normalized > group_size:
        raise ValueError(f"valid_items must be at most {group_size}")


def plan_group_primitive(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> GroupLoweringPlan:
    """Resolve one scalar block reduction to a typed CUB plan."""

    if not isinstance(call, GroupPrimitiveCall):
        raise TypeError("call must be a GroupPrimitiveCall")
    operation = call.operation
    _validate_static_valid_items(operation.valid_items)
    resolution = resolve_thread_group(call.group, launch)
    if resolution.unsupported is not None:
        return _unsupported_plan(call, resolution)
    group = resolution.group
    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    _validate_static_valid_items(
        operation.valid_items,
        group_size=launch.exact_block_threads,
    )
    implementation = make_block_reduce_spec(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        operation=operation.operation,
        binary_op=operation.binary_op,
        algorithm=operation.algorithm,
        valid_items=operation.has_valid_items,
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=group,
        implementation=implementation,
        participation=ParticipationContract(
            group_kind="block",
            exact_group_size=launch.exact_block_threads,
            exact_block_dim=launch.exact_block_dim,
            complete_membership=True,
            converged_entry=True,
            uniform_arguments=("valid_items",) if operation.has_valid_items else (),
            valid_member_selection=(
                "first valid_items block members" if operation.has_valid_items else None
            ),
        ),
        result=ResultContract(dtype=operation.dtype),
        synchronization=SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=SynchronizationScope.BLOCK,
        ),
        temp_storage=TempStorageContract(
            ownership=StorageOwnership.IMPLEMENTATION,
        ),
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_reduce.cuh",
            cpp_class="cub::BlockReduce",
            method=implementation.method_name,
        ),
    )


__all__ = [
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupOperandKind",
    "GroupPrimitiveCall",
    "GroupReduceAlgorithm",
    "GroupReduceOperation",
    "GroupReduceOperator",
    "GroupReduceSemantics",
    "ImplementationProvenance",
    "ParticipationContract",
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
    "make_group_primitive_call",
    "plan_group_primitive",
    "resolve_thread_group",
]

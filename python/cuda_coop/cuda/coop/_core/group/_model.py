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

from .._symbols import semantic_token
from ..launch import Dim3
from ..thread_group import ThreadGroup

if TYPE_CHECKING:
    from ..block.reduce import BlockReduceSpec
    from ._dispatch import GroupOperationSemantics


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
class GroupPrimitiveCall:
    """One canonical group reduction call before launch resolution."""

    group: ThreadGroup
    operation: GroupOperationSemantics
    source: str = field(default="canonical", compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("GroupPrimitiveCall group must be a ThreadGroup")
        from ._dispatch import _is_group_operation

        if not _is_group_operation(self.operation):
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


__all__ = [
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupOperandKind",
    "GroupPrimitiveCall",
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
]

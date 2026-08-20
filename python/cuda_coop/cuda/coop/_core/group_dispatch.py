# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed planning for block-wide direct Load and Store."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._bindings import ArgumentBinding, BindingKind
from ._symbols import semantic_token
from .block.load_store import (
    BlockLoadStoreAlgorithm,
    BlockLoadStoreKind,
    BlockLoadStoreSpec,
    make_block_load_spec,
    make_block_store_spec,
)
from .launch import Dim3, LaunchFacts
from .thread_group import ThreadGroup, ThreadHierarchy


class GroupLoweringTarget(str, Enum):
    CUB_BLOCK = "cub_block"
    UNSUPPORTED = "unsupported"


GroupLoadStoreKind = BlockLoadStoreKind
GroupLoadStoreAlgorithm = BlockLoadStoreAlgorithm


class ResultVisibility(str, Enum):
    PER_MEMBER = "per_member"


class ResultOwnership(str, Enum):
    EACH_MEMBER = "each_member"


class GroupOperandKind(str, Enum):
    ARRAY = "array"


class StorageOwnership(str, Enum):
    IMPLEMENTATION = "implementation"


class SynchronizationScope(str, Enum):
    BLOCK = "block"


class UnsupportedReasonCode(str, Enum):
    MISSING_EXACT_BLOCK_DIM = "missing_exact_block_dim"
    UNVERIFIED_EXACT_BLOCK_DIM = "unverified_exact_block_dim"
    GROUP_KIND = "group_kind"


@dataclass(frozen=True, eq=False)
class GroupLoadStoreSemantics:
    """Backend-neutral semantics of one block-wide direct Load or Store."""

    kind: GroupLoadStoreKind
    dtype: Any
    items_per_thread: int
    algorithm: GroupLoadStoreAlgorithm = GroupLoadStoreAlgorithm.DIRECT
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    oob_default: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    offset: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GroupLoadStoreKind(self.kind))
        object.__setattr__(self, "algorithm", GroupLoadStoreAlgorithm(self.algorithm))
        if (
            not isinstance(self.items_per_thread, int)
            or isinstance(self.items_per_thread, bool)
            or self.items_per_thread <= 0
        ):
            raise ValueError("items_per_thread must be a positive integer")
        for name in ("valid_items", "oob_default", "offset"):
            if not isinstance(getattr(self, name), ArgumentBinding):
                raise TypeError(f"{name} must be an ArgumentBinding")
        if (
            self.kind is GroupLoadStoreKind.STORE
            and self.oob_default.kind is not BindingKind.OMITTED
        ):
            raise ValueError("oob_default is valid only for group load")
        if (
            self.oob_default.kind is not BindingKind.OMITTED
            and self.valid_items.kind is BindingKind.OMITTED
        ):
            raise ValueError("oob_default requires valid_items")

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def has_oob_default(self) -> bool:
        return self.oob_default.kind is not BindingKind.OMITTED

    @property
    def has_offset(self) -> bool:
        return self.offset.kind is not BindingKind.OMITTED

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            f"group_{self.kind.value}",
            semantic_token(self.dtype),
            self.items_per_thread,
            self.algorithm.value,
            self.valid_items,
            self.oob_default,
            self.offset,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupLoadStoreSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupPrimitiveCall:
    group: ThreadGroup
    operation: GroupLoadStoreSemantics
    source: str = field(default="canonical", compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("GroupPrimitiveCall group must be a ThreadGroup")
        if not isinstance(self.operation, GroupLoadStoreSemantics):
            raise TypeError("GroupPrimitiveCall operation must be Load or Store")

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
    group_kind: str
    exact_group_size: int
    exact_block_dim: Dim3
    complete_membership: bool
    contiguous: bool
    aligned: bool
    converged_entry: bool
    uniform_arguments: tuple[str, ...] = ()
    valid_member_selection: str | None = None


@dataclass(frozen=True)
class LogicalResultContract:
    name: str
    dtype: Any
    visibility: ResultVisibility
    ownership: ResultOwnership
    operand_kind: GroupOperandKind
    items_per_member: int

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.name,
            semantic_token(self.dtype),
            self.visibility.value,
            self.ownership.value,
            self.operand_kind.value,
            self.items_per_member,
        )

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class ResultContract:
    values: tuple[LogicalResultContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("result contract requires at least one result")

    @property
    def primary(self) -> LogicalResultContract:
        return self.values[0]

    @property
    def visibility(self) -> ResultVisibility:
        return self.primary.visibility

    @property
    def result_items_per_thread(self) -> int:
        return self.primary.items_per_member


@dataclass(frozen=True)
class SynchronizationContract:
    converged_entry: bool
    storage_reuse_barrier: SynchronizationScope


@dataclass(frozen=True)
class TempStorageContract:
    ownership: StorageOwnership
    address_space: str | None = None
    cpp_type: str | None = None
    instances: int | None = None
    instance_index: str | None = None
    exact_layout_required: bool = False


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
    group: ThreadGroup
    unsupported: UnsupportedReason | None = None

    def require_supported(self) -> ThreadGroup:
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self.group


@dataclass(frozen=True, eq=False)
class GroupLoweringPlan:
    target: GroupLoweringTarget
    call: GroupPrimitiveCall
    resolved_group: ThreadGroup
    implementation: BlockLoadStoreSpec | None
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
        result_required = self.call.operation.kind is GroupLoadStoreKind.LOAD
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
        return self.resolved_group.semantic_key, self.call.operation.semantic_key

    @property
    def artifact_key(self) -> tuple[Any, ...]:
        if self.unsupported is not None:
            return (
                self.target.value,
                "unsupported",
                self.unsupported.code.value,
            )
        assert self.implementation is not None
        assert self.participation is not None
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
    operation: GroupLoadStoreSemantics,
    *,
    source: str = "canonical",
) -> GroupPrimitiveCall:
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
                "block operation requires exact block dimensions",
            ),
        )
    if not launch.is_verified("exact_block_dim"):
        return ThreadGroupResolution(
            group,
            UnsupportedReason(
                UnsupportedReasonCode.UNVERIFIED_EXACT_BLOCK_DIM,
                "block operation requires compiler-verified exact block dimensions",
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


def _validate_static_integer_binding(
    binding: ArgumentBinding,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if binding.kind is not BindingKind.STATIC:
        return
    value = binding.value
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def plan_group_primitive(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> GroupLoweringPlan:
    """Resolve one direct block Load or Store to a typed CUB plan."""

    if not isinstance(call, GroupPrimitiveCall):
        raise TypeError("call must be a GroupPrimitiveCall")
    operation = call.operation
    _validate_static_integer_binding(
        operation.valid_items,
        name="valid_items",
        minimum=0,
    )
    _validate_static_integer_binding(operation.offset, name="offset", minimum=0)
    resolution = resolve_thread_group(call.group, launch)
    if resolution.unsupported is not None:
        return _unsupported_plan(call, resolution)
    group = resolution.group
    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    tile_items = launch.exact_block_threads * operation.items_per_thread
    _validate_static_integer_binding(
        operation.valid_items,
        name="valid_items",
        maximum=tile_items,
    )
    make_spec = (
        make_block_load_spec
        if operation.kind is GroupLoadStoreKind.LOAD
        else make_block_store_spec
    )
    implementation = make_spec(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=operation.items_per_thread,
        algorithm=operation.algorithm,
        valid_items=operation.has_valid_items,
        oob_default=operation.has_oob_default,
        include_pointer_offset=operation.has_offset,
    )
    uniform_arguments = tuple(
        name
        for name, binding in (
            ("valid_items", operation.valid_items),
            ("oob_default", operation.oob_default),
            ("offset", operation.offset),
        )
        if binding.kind is not BindingKind.OMITTED
    )
    result = None
    if operation.kind is GroupLoadStoreKind.LOAD:
        result = ResultContract(
            (
                LogicalResultContract(
                    name="value",
                    dtype=operation.dtype,
                    visibility=ResultVisibility.PER_MEMBER,
                    ownership=ResultOwnership.EACH_MEMBER,
                    operand_kind=GroupOperandKind.ARRAY,
                    items_per_member=operation.items_per_thread,
                ),
            )
        )
    title = operation.kind.value.title()
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
            contiguous=True,
            aligned=True,
            converged_entry=True,
            uniform_arguments=uniform_arguments,
            valid_member_selection=(
                "first valid_items tile elements" if operation.has_valid_items else None
            ),
        ),
        result=result,
        synchronization=SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=SynchronizationScope.BLOCK,
        ),
        temp_storage=TempStorageContract(
            ownership=StorageOwnership.IMPLEMENTATION,
        ),
        provenance=ImplementationProvenance(
            library="CUB",
            header=f"cub/block/block_{operation.kind.value}.cuh",
            cpp_class=f"cub::Block{title}",
            method=title,
        ),
    )


__all__ = [
    "GroupLoadStoreAlgorithm",
    "GroupLoadStoreKind",
    "GroupLoadStoreSemantics",
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupOperandKind",
    "GroupPrimitiveCall",
    "ImplementationProvenance",
    "LogicalResultContract",
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

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-family participation, result, storage, and failure contracts.

Family planners call these helpers only after resolving a static thread group.
The helpers centralize cache-relevant contracts without owning any primitive's
semantic choices or a backend's compiler lifecycle.
"""

from __future__ import annotations

from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    GroupTopologyContract,
    ParticipationContract,
    ResultContract,
    StorageOwnership,
    SynchronizationContract,
    SynchronizationScope,
    TempStorageContract,
    UnsupportedReason,
    UnsupportedReasonCode,
)


def _unsupported(
    call: GroupPrimitiveCall,
    resolved_group: ThreadGroup,
    code: UnsupportedReasonCode,
    message: str,
) -> GroupLoweringPlan:
    return GroupLoweringPlan(
        target=GroupLoweringTarget.UNSUPPORTED,
        call=call,
        resolved_group=resolved_group,
        implementation=None,
        topology=None,
        participation=None,
        result=None,
        synchronization=None,
        temp_storage=None,
        provenance=None,
        unsupported=UnsupportedReason(code=code, message=message),
    )


def _group_topology(
    resolved_group: ThreadGroup,
    launch: LaunchFacts,
) -> GroupTopologyContract:
    """Describe group instances without depending on a primitive family."""

    group_size = resolved_group.static_size
    if group_size is None:
        raise ValueError("group contracts require a static group size")

    block_threads = launch.exact_block_threads
    kind = resolved_group.kind
    if kind == "thread":
        if block_threads is None:
            raise ValueError("thread contracts require exact block dimensions")
        instances = block_threads
        index = "linear_thread_rank"
        thread_rank = "0"
        execution_scope = SynchronizationScope.NONE
    elif kind == "warp":
        if block_threads is None:
            raise ValueError("warp contracts require exact block dimensions")
        if block_threads % group_size != 0:
            raise ValueError("group width must divide the enclosing block size")
        instances = block_threads // group_size
        index = f"linear_thread_rank / {group_size}"
        thread_rank = f"linear_thread_rank % {group_size}"
        execution_scope = SynchronizationScope.WARP
    elif kind == "threads_within_warp":
        if block_threads is None:
            raise ValueError("warp contracts require exact block dimensions")
        mapping = resolved_group.mapping
        assert mapping is not None
        groups_per_warp = resolved_group.groups_per_parent
        assert groups_per_warp is not None
        instances = (block_threads // 32) * groups_per_warp
        if mapping.exhaustive:
            index = f"linear_thread_rank / {group_size}"
            thread_rank = f"linear_thread_rank % {group_size}"
        else:
            index = (
                f"(linear_thread_rank / 32) * {groups_per_warp} + "
                f"((linear_thread_rank % 32) / {group_size})"
            )
            thread_rank = f"(linear_thread_rank % 32) % {group_size}"
        execution_scope = SynchronizationScope.WARP
    elif kind == "warps_within_block":
        if block_threads is None:
            raise ValueError("mapped block contracts require exact block dimensions")
        mapping = resolved_group.mapping
        assert mapping is not None
        groups_per_block = resolved_group.groups_per_parent
        assert groups_per_block is not None
        instances = groups_per_block
        if mapping.exhaustive:
            index = f"linear_thread_rank / {group_size}"
            thread_rank = f"linear_thread_rank % {group_size}"
        else:
            index = f"(linear_thread_rank / 32) / {mapping.count}"
            thread_rank = (
                f"((linear_thread_rank / 32) % {mapping.count}) * 32 + "
                "(linear_thread_rank % 32)"
            )
        execution_scope = (
            SynchronizationScope.WARP
            if group_size == 32
            else SynchronizationScope.GROUP
        )
    elif kind == "block":
        instances = 1
        index = "cta"
        thread_rank = "linear_thread_rank"
        execution_scope = SynchronizationScope.BLOCK
    elif kind == "cluster":
        instances = 1
        index = "cluster"
        thread_rank = "group.thread_rank()"
        execution_scope = SynchronizationScope.GROUP
    elif kind == "grid":
        instances = 1
        index = "grid"
        thread_rank = "group.thread_rank()"
        execution_scope = SynchronizationScope.GROUP
    else:
        instances = resolved_group.groups_per_parent or 1
        index = "group.rank(parent)"
        thread_rank = "group.thread_rank()"
        execution_scope = SynchronizationScope.GROUP

    return GroupTopologyContract(
        group_kind=kind,
        logical_width=group_size,
        instances=instances,
        instance_index=index,
        thread_rank=thread_rank,
        execution_scope=execution_scope,
    )


def _contracts(
    resolved_group: ThreadGroup,
    launch: LaunchFacts,
    *,
    result: ResultContract | None,
    storage_ownership: StorageOwnership,
    cpp_type: str | None,
    storage_sharing: str | None = None,
    requested_size_in_bytes: int | None = None,
    requested_alignment: int | None = None,
    auto_sync: bool | None = None,
    uniform_arguments: tuple[str, ...] = (),
    valid_member_selection: str | None = None,
    argument_preconditions: tuple[ArgumentPrecondition, ...] = (),
) -> tuple[
    GroupTopologyContract,
    ParticipationContract,
    SynchronizationContract,
    TempStorageContract,
]:
    group_size = resolved_group.static_size
    assert group_size is not None
    topology = _group_topology(resolved_group, launch)
    storage_ownership = StorageOwnership(storage_ownership)
    storage_free = storage_ownership is StorageOwnership.NONE
    if auto_sync is None:
        auto_sync = not storage_free
    barrier = (
        SynchronizationScope.NONE
        if storage_free or not auto_sync
        else topology.execution_scope
    )
    return (
        topology,
        ParticipationContract(
            group_kind=resolved_group.kind,
            exact_group_size=group_size,
            exact_block_dim=launch.exact_block_dim,
            complete_membership=resolved_group.complete_membership is not False,
            contiguous=True,
            aligned=True,
            converged_entry=True,
            complete_parent_partition=(
                resolved_group.kind == "warp"
                or resolved_group.complete_membership is True
            ),
            uniform_arguments=uniform_arguments,
            valid_member_selection=valid_member_selection,
            argument_preconditions=argument_preconditions,
        ),
        SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=barrier,
        ),
        TempStorageContract(
            ownership=storage_ownership,
            address_space=None if storage_free else "shared",
            cpp_type=cpp_type,
            instances=(
                None
                if storage_ownership is StorageOwnership.NONE
                else topology.instances
            ),
            instance_index=(
                None
                if storage_ownership is StorageOwnership.NONE
                else topology.instance_index
            ),
            exact_layout_required=storage_ownership is StorageOwnership.CALLER,
            sharing=storage_sharing,
            requested_size_in_bytes=requested_size_in_bytes,
            requested_alignment=requested_alignment,
            auto_sync=auto_sync,
        ),
    )


def _cub_warp_width(group: ThreadGroup) -> int:
    """Return a CUB-legal physical or logical warp width."""

    if group.kind == "warp":
        return 32
    if group.kind != "threads_within_warp":
        raise ValueError("CUB warp primitives require a warp-based group")
    width = group.static_size
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or width > 32
        or width & (width - 1)
        or 32 % width != 0
    ):
        raise ValueError(
            "CUB-backed logical-warp operations require a power-of-two group "
            "width in [1, 32] that divides the 32-thread physical warp; "
            f"got {width!r}"
        )
    return width


def _unsupported_cub_warp_width(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
) -> tuple[int | None, GroupLoweringPlan | None]:
    try:
        return _cub_warp_width(resolved), None
    except ValueError as exc:
        return None, _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            str(exc),
        )


__all__ = []

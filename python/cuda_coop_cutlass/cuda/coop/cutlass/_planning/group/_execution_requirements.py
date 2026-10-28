# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe how a resolved group must participate and use scratch storage.

Operation planners call these helpers after choosing a group shape. The
result tells a backend how many group instances need storage, which
threads belong to each instance, and where storage reuse needs a barrier.
For example, several logical warps in one block need separate storage
instances even when they use the same primitive.

These records state execution requirements. They do not prove that a
kernel reaches a call uniformly or insert synchronization themselves.
Backend lowering must honor the plan when it creates the device call.
"""

from __future__ import annotations

from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._model import (
    ArgumentPrecondition,
    GroupExecutionRequirements,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    GroupTopologyRequirements,
    ParticipationRequirements,
    StorageOwnership,
    SynchronizationRequirements,
    SynchronizationScope,
    TempStorageRequirements,
    UnsupportedReason,
    UnsupportedReasonCode,
)


def _unsupported(
    call: GroupPrimitiveCall,
    resolved_group: ThreadGroup,
    code: UnsupportedReasonCode,
    message: str,
) -> GroupLoweringPlan:
    """Return a plan containing the call and the failed requirement.

    Leave implementation and execution requirements unset. Callers can inspect
    or report the failure without mistaking it for a usable device plan.
    """

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
) -> GroupTopologyRequirements:
    """Describe group instances and each thread's rank within an instance.

    This calculation is shared by primitive families. For a warp-based group,
    linear thread rank selects the instance by division and the rank within
    it by remainder. Whole blocks, clusters, and grids use their enclosing
    execution scope.

    Parameters
    ----------
    resolved_group : ThreadGroup
        Group with a known thread count. Mapped groups must fit the block.
    launch : LaunchFacts
        Exact block size needed to count thread or warp-based instances.

    Returns
    -------
    GroupTopologyRequirements
        Group width, instance count, rank expressions, and execution scope.
        Expressions describe the rank calculation for backend lowering.

    Raises
    ------
    ValueError
        The group size is unknown, required block dimensions are missing,
        or a warp-based group does not divide the block.
    """

    group_size = resolved_group.static_size
    if group_size is None:
        raise ValueError("group topology requires a static group size")

    block_threads = launch.exact_block_threads
    kind = resolved_group.kind
    if kind == "thread":
        if block_threads is None:
            raise ValueError("thread topology requires exact block dimensions")
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
        if resolved_group.complete_membership is True:
            index = f"linear_thread_rank / {group_size}"
            thread_rank = f"linear_thread_rank % {group_size}"
        else:
            # Restart the mapping inside each physical warp. Its trailing
            # lanes do not form another complete logical group.
            index = (
                f"(linear_thread_rank / 32) * {groups_per_warp} + "
                f"((linear_thread_rank % 32) / {group_size})"
            )
            thread_rank = f"(linear_thread_rank % 32) % {group_size}"
        execution_scope = SynchronizationScope.WARP
    elif kind == "warps_within_block":
        if block_threads is None:
            raise ValueError(
                "mapped block topology requires exact block dimensions"
            )
        mapping = resolved_group.mapping
        assert mapping is not None
        groups_per_block = resolved_group.groups_per_parent
        assert groups_per_block is not None
        instances = groups_per_block
        if resolved_group.complete_membership is True:
            index = f"linear_thread_rank / {group_size}"
            thread_rank = f"linear_thread_rank % {group_size}"
        else:
            # Count whole warps before locating a group. The membership
            # contract excludes remainder warps outside complete groups.
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

    return GroupTopologyRequirements(
        group_kind=kind,
        logical_width=group_size,
        instances=instances,
        instance_index=index,
        thread_rank=thread_rank,
        execution_scope=execution_scope,
    )


def _build_execution_requirements(
    resolved_group: ThreadGroup,
    launch: LaunchFacts,
    *,
    storage_ownership: StorageOwnership,
    cpp_type: str | None,
    storage_sharing: str | None = None,
    requested_size_in_bytes: int | None = None,
    requested_alignment: int | None = None,
    auto_sync: bool | None = None,
    uniform_arguments: tuple[str, ...] = (),
    valid_member_selection: str | None = None,
    argument_preconditions: tuple[ArgumentPrecondition, ...] = (),
) -> GroupExecutionRequirements:
    """Build participation, synchronization, and storage requirements.

    A resolved group's topology determines both the storage instance index
    and the scope of a reuse barrier. Storage-free implementations need no
    reuse barrier. For implementations with storage, ``auto_sync`` defaults
    to true; a false value leaves storage reuse synchronization to the caller.

    The participation record requires contiguous, aligned members to enter
    together. Operation planners add argument-specific requirements, such as
    which counts must be uniform. These are requirements for later lowering
    and execution, not runtime checks performed by this helper.

    Parameters
    ----------
    resolved_group : ThreadGroup
        Group with a known size, already accepted by the operation planner.
    launch : LaunchFacts
        Exact launch dimensions used to describe participation and instances.
    storage_ownership : StorageOwnership
        Whether the primitive uses no storage, storage supplied by its
        implementation, or storage supplied by the caller.
    cpp_type : str or None
        C++ storage type selected by the implementation, when applicable.
    storage_sharing : str, optional
        Sharing policy to carry into the storage requirements.
    requested_size_in_bytes : int, optional
        Caller-requested storage capacity to check against the actual layout.
    requested_alignment : int, optional
        Caller-requested storage alignment to check against the actual layout.
    auto_sync : bool, optional
        Whether lowering should arrange a barrier before storage reuse.
        Omission enables it for implementations that use storage.
    uniform_arguments : tuple of str, optional
        Argument names whose values must agree across participating threads.
    valid_member_selection : str, optional
        Description of which members or data elements the operation uses.
        Reduce uses the first ``valid_items`` members; a guarded load uses
        the first ``valid_items`` tile elements.
    argument_preconditions : tuple of ArgumentPrecondition, optional
        Additional bounds or other requirements on operation arguments.

    Returns
    -------
    GroupExecutionRequirements
        Topology, participation, synchronization, and temporary-storage
        requirements. Caller-owned storage requires an exact
        layout check; this helper does not calculate the compiled layout.
    """

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
    return GroupExecutionRequirements(
        topology=topology,
        participation=ParticipationRequirements(
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
        synchronization=SynchronizationRequirements(
            converged_entry=True,
            storage_reuse_barrier=barrier,
        ),
        temp_storage=TempStorageRequirements(
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
    """Select a width that CUB's warp primitives can represent.

    Physical warps use 32 threads. A logical warp must have a power-of-two
    width from 1 through 32 and divide its physical warp. Raise ``ValueError``
    for a different group kind or width. Group resolution has already checked
    that the block contains only complete 32-thread warps.
    """

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
    """Return the CUB width or an unsupported plan for an invalid group.

    Translate the width check's ``ValueError`` into the same structured
    failure that other primitive-planning checks return.
    """

    try:
        return _cub_warp_width(resolved), None
    except ValueError as exc:
        return None, _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            str(exc),
        )


__all__ = [
    "_build_execution_requirements",
    "_group_topology",
    "_unsupported",
    "_unsupported_cub_warp_width",
]

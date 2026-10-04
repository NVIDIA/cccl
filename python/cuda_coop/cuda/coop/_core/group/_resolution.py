# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve group descriptions using the kernel's launch facts.

A descriptor can name the current block before its dimensions are known.
Resolution fills in the exact dimensions needed for that group or for a
query about an enclosing group. It also checks complete physical warps
and mapped-group partitions.

Missing facts and unsupported shapes produce a structured reason that an
operation planner can report. Contradictory explicit dimensions raise an
error. Cluster and grid resolution also require verified cluster-launch state.
Operation dispatch adds capability checks specific to the operation, such as
cooperative-launch support for grid operations.
"""

from __future__ import annotations

from ..launch import LaunchFacts
from ..thread_group import (
    COMPLETE_WARP_GROUP_KINDS,
    ThreadGroup,
    ThreadHierarchy,
    normalize_thread_level,
)
from ._execution_requirements import _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupPrimitiveCall,
    ThreadGroupLaunchResolution,
    UnsupportedReason,
    UnsupportedReasonCode,
)

_THREAD_LEVEL_ORDER = {
    "thread": 0,
    "warp": 1,
    "block": 2,
    "cluster": 3,
    "grid": 4,
}

_MAPPED_PARENT_LEVEL = {
    "threads_within_warp": "warp",
    "warps_within_block": "block",
}


def _resolution_failure(
    group: ThreadGroup,
    code: UnsupportedReasonCode,
    message: str,
) -> ThreadGroupLaunchResolution:
    """Keep the requested group with a reason that prevents its resolution."""

    return ThreadGroupLaunchResolution(
        group=group,
        unsupported=UnsupportedReason(code=code, message=message),
    )


def resolve_thread_group(
    group: ThreadGroup,
    launch: LaunchFacts,
    *,
    through_level: str | None = None,
) -> ThreadGroupLaunchResolution:
    """Fill a group's required hierarchy from exact launch facts.

    The group's own level determines how much hierarchy an operation needs.
    ``through_level`` can request more, for example when a group query asks
    for a count at an enclosing level. A mapped group needs the dimensions
    of its physical parent.

    Exact dimensions remain distinct from upper bounds. Cluster and grid
    resolution require verified cluster-launch state. A verified non-cluster
    launch lets a missing cluster shape resolve to one block. Each grid axis's
    block count must be divisible by the cluster's block count on that axis.
    The resolved hierarchy stores the resulting cluster counts.

    Warp-based groups require complete 32-thread physical warps in the
    block. A mapped group's count must fit its parent, and an exhaustive
    mapping must divide the parent's unit count exactly.

    Parameters
    ----------
    group : ThreadGroup
        Descriptor to resolve. Any dimensions already present must agree
        with the required launch dimensions.
    launch : LaunchFacts
        Exact dimensions and capability evidence for this compilation.
    through_level : str, optional
        Enclosing level needed by a group query. Primitive planners omit
        this argument when the group's own level is sufficient.

    Returns
    -------
    ThreadGroupLaunchResolution
        The resolved group, or the original group with an unsupported reason.
        Missing launch facts and unsupported partitions use this result
        instead of raising an exception. A request that needs only the thread
        level returns the group unchanged; it needs no launch dimensions.

    Raises
    ------
    TypeError
        The group or launch argument has the wrong type.
    ValueError
        The requested level is invalid or an existing group dimension
        contradicts the exact launch.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError("group must be a ThreadGroup")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")

    group_level = _MAPPED_PARENT_LEVEL.get(group.kind, group.kind)
    if through_level is not None:
        through_level = normalize_thread_level(
            through_level,
            scope="resolve_thread_group",
            feature="through_level",
        )
    required_level = max(
        (group_level, through_level or group_level),
        key=_THREAD_LEVEL_ORDER.__getitem__,
    )
    needs_complete_warp = group.kind in COMPLETE_WARP_GROUP_KINDS
    if required_level == "thread":
        return ThreadGroupLaunchResolution(group)

    exact_block_dim = launch.exact_block_dim
    if exact_block_dim is None:
        return _resolution_failure(
            group,
            UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM,
            "group operation requires exact block dimensions; max_block_dim "
            "is only an upper bound",
        )
    assert group.hierarchy is not None
    if (
        group.hierarchy.block_dim is not None
        and group.hierarchy.block_dim != exact_block_dim
    ):
        raise ValueError(
            f"group block dimensions {group.hierarchy.block_dim!r} do not "
            f"match the exact kernel launch dimensions {exact_block_dim!r}",
        )
    needs_cluster = (
        _THREAD_LEVEL_ORDER[required_level] >= _THREAD_LEVEL_ORDER["cluster"]
    )
    exact_cluster_dim = launch.exact_cluster_dim if needs_cluster else None
    if needs_cluster:
        cluster_launch_verified = launch.is_verified("cluster_launch")
        if exact_cluster_dim is None:
            if (
                launch.cluster_launch is not False
                or not cluster_launch_verified
            ):
                return _resolution_failure(
                    group,
                    UnsupportedReasonCode.LAUNCH_CAPABILITY,
                    "cluster and grid group operations require exact static "
                    "cluster dimensions, or a backend-verified "
                    "non-cluster launch",
                )
            exact_cluster_dim = (1, 1, 1)
        elif launch.cluster_launch is None or not cluster_launch_verified:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "cluster and grid group operations require backend-verified "
                "cluster launch state",
            )
        elif (
            exact_cluster_dim != (1, 1, 1) and launch.cluster_launch is not True
        ):
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "multi-block cluster operations require verified "
                "cluster launch capability",
            )

    hierarchy_grid_dim = None
    needs_grid = required_level == "grid"
    if needs_grid:
        exact_grid_dim = launch.exact_grid_dim
        if exact_grid_dim is None:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "grid group operations require exact static grid dimensions",
            )
        assert exact_cluster_dim is not None
        if any(
            grid_extent % cluster_extent != 0
            for grid_extent, cluster_extent in zip(
                exact_grid_dim,
                exact_cluster_dim,
            )
        ):
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "physical CTA grid dimensions must be divisible by "
                "the cluster dimensions",
            )
        hierarchy_grid_dim = tuple(
            grid_extent // cluster_extent
            for grid_extent, cluster_extent in zip(
                exact_grid_dim,
                exact_cluster_dim,
            )
        )

    resolved_hierarchy = ThreadHierarchy._resolved(
        block_dim=exact_block_dim,
        cluster_dim=(exact_cluster_dim if needs_cluster else None),
        grid_dim=hierarchy_grid_dim,
    )
    if (
        needs_cluster
        and group.hierarchy.cluster_dim is not None
        and (group.hierarchy.cluster_dim != resolved_hierarchy.cluster_dim)
    ):
        raise ValueError(
            f"group cluster dimensions {group.hierarchy.cluster_dim!r} do not "
            f"match exact launch dimensions {resolved_hierarchy.cluster_dim!r}"
        )
    if (
        needs_grid
        and group.hierarchy.grid_dim is not None
        and (group.hierarchy.grid_dim != resolved_hierarchy.grid_dim)
    ):
        raise ValueError(
            f"group grid dimensions {group.hierarchy.grid_dim!r} do not match "
            f"exact hierarchy dimensions {resolved_hierarchy.grid_dim!r}"
        )
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    if needs_complete_warp and block_threads % 32 != 0:
        return _resolution_failure(
            group,
            UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
            "physical-warp operation requires complete 32-thread warps and "
            "every physical warp in the enclosing CTA to be complete; got "
            f"{block_threads} block threads",
        )
    queries_warps_as_constituents = (
        through_level == "warp"
        and _THREAD_LEVEL_ORDER[group_level] > _THREAD_LEVEL_ORDER["warp"]
    )
    # A containing group's query can count a partial final warp, but needs
    # at least one complete warp. A physical-warp operation itself still
    # requires every warp in the block to be complete, as checked above.
    if queries_warps_as_constituents and block_threads < 32:
        return _resolution_failure(
            group,
            UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
            "physical-Warp hierarchy queries require at least one complete "
            f"32-thread Warp; got {block_threads} block threads",
        )
    if group.mapping is not None:
        parent_units = (
            32 if group.kind == "threads_within_warp" else block_threads // 32
        )
        if group.mapping.count > parent_units:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.GROUP_KIND,
                "mapped group count exceeds the resolved parent unit count",
            )
        if group.mapping.exhaustive and parent_units % group.mapping.count != 0:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.GROUP_KIND,
                "exhaustive mapped group count must divide "
                "the resolved parent unit count",
            )
    resolved = group.with_hierarchy(
        resolved_hierarchy,
        source="launch_facts",
    )
    return ThreadGroupLaunchResolution(resolved)


def _resolve_group(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> tuple[ThreadGroup, GroupLoweringPlan | None]:
    """Resolve a primitive's group and convert failure to an operation plan.

    Return ``(resolved_group, None)`` on success. Otherwise return the group
    with an unsupported plan that retains the call and the resolver's reason.
    Dispatch returns this plan directly, so a resolution failure has the same
    form as an unsupported result from a family planner.
    """

    resolution = resolve_thread_group(call.group, launch)
    if resolution.unsupported is None:
        return resolution.group, None
    return resolution.group, _unsupported(
        call,
        resolution.group,
        resolution.unsupported.code,
        resolution.unsupported.message,
    )


__all__ = [
    "_resolve_group",
    "resolve_thread_group",
]

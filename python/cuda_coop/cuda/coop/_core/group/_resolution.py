# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve symbolic thread groups against compiler-provided launch facts.

Resolution is shared by every primitive family and records typed failures when
the launch cannot prove a safe static group. It does not choose an algorithm,
construct an artifact, or inspect backend compiler state.
"""

from __future__ import annotations

from ..launch import LaunchFacts
from ..thread_group import (
    COMPLETE_WARP_GROUP_KINDS,
    ThreadGroup,
    ThreadHierarchy,
    normalize_thread_level,
)
from ._contracts import _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupPrimitiveCall,
    ThreadGroupResolution,
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
) -> ThreadGroupResolution:
    return ThreadGroupResolution(
        group=group,
        unsupported=UnsupportedReason(code=code, message=message),
    )


def resolve_thread_group(
    group: ThreadGroup,
    launch: LaunchFacts,
    *,
    through_level: str | None = None,
) -> ThreadGroupResolution:
    """Resolve a group against exact launch facts through a hierarchy level.

    ``through_level`` requests the enclosing hierarchy needed by group queries.
    Collective planners omit it because the group's own level is sufficient.
    Exact dimensions remain distinct from upper bounds, and cluster state must
    be verified before a missing cluster extent can be treated as one block.
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
    needs_complete_warp = (
        group.kind in COMPLETE_WARP_GROUP_KINDS or through_level == "warp"
    )
    if required_level == "thread":
        return ThreadGroupResolution(group)

    exact_block_dim = launch.exact_block_dim
    if exact_block_dim is None:
        return _resolution_failure(
            group,
            UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM,
            "group operation requires exact block dimensions; max_block_dim "
            "is only an upper bound",
        )
    assert group.hierarchy is not None
    if group.hierarchy.block_dim is not None:
        if group.hierarchy.block_dim != exact_block_dim:
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
            if launch.cluster_launch is not False or not cluster_launch_verified:
                return _resolution_failure(
                    group,
                    UnsupportedReasonCode.LAUNCH_CAPABILITY,
                    "cluster and grid group operations require exact static "
                    "cluster dimensions, or a backend-verified non-cluster launch",
                )
            exact_cluster_dim = (1, 1, 1)
        elif launch.cluster_launch is None or not cluster_launch_verified:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "cluster and grid group operations require backend-verified "
                "cluster launch state",
            )
        elif exact_cluster_dim != (1, 1, 1) and launch.cluster_launch is not True:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "multi-block cluster operations require verified cluster launch "
                "capability",
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
                "physical CTA grid dimensions must be divisible by the cluster "
                "dimensions",
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
    if needs_complete_warp and launch.exact_block_threads % 32 != 0:  # type: ignore[operator]
        return _resolution_failure(
            group,
            UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
            "physical-warp operation requires complete 32-thread warps and "
            "every physical warp in the enclosing CTA to be complete; got "
            f"{launch.exact_block_threads} block threads",
        )
    if group.mapping is not None:
        parent_units = (
            32
            if group.kind == "threads_within_warp"
            else launch.exact_block_threads // 32  # type: ignore[operator]
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
                "exhaustive mapped group count must divide the resolved parent "
                "unit count",
            )
    resolved = group.with_hierarchy(
        resolved_hierarchy,
        source="launch_facts",
    )
    return ThreadGroupResolution(resolved)


def _resolve_group(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> tuple[ThreadGroup, GroupLoweringPlan | None]:
    resolution = resolve_thread_group(call.group, launch)
    if resolution.unsupported is None:
        return resolution.group, None
    return resolution.group, _unsupported(
        call,
        resolution.group,
        resolution.unsupported.code,
        resolution.unsupported.message,
    )


__all__ = ["resolve_thread_group"]

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS bindings for shared CUDA thread-hierarchy descriptors."""

from __future__ import annotations

from typing import Any

from cuda.coop._core import (
    COMPLETE_WARP_GROUP_KINDS,
    PHYSICAL_GROUP_KINDS,
    LaunchFacts,
    ThreadHierarchy,
    UnsupportedReasonCode,
    cpp_level_expr,
    make_thread_group,
    normalize_thread_level,
    render_group_decl,
    render_group_decl_lines,
    render_hierarchy_decl,
    resolve_thread_group,
)
from cuda.coop._core.thread_group import ThreadGroup as PortableThreadGroup

_ROOT_SCOPE = __name__.rsplit(".", 1)[0]
Hierarchy = ThreadHierarchy


class ThreadGroup(PortableThreadGroup):
    """Shared group descriptor with CUTLASS provider operations attached."""

    def rank(self, level: str = "thread") -> Any:
        """Return this group's rank relative to another hierarchy level."""

        return self.rank_as(None, level)

    def count(self, level: str = "thread") -> Any:
        """Return this group's count relative to another hierarchy level."""

        return self.count_as(None, level)

    def rank_as(self, dtype: Any = None, level: str = "thread") -> Any:
        level = normalize_thread_level(
            level,
            scope=_ROOT_SCOPE,
            feature="ThreadGroup.rank",
        )
        group = _validate_query_launch(
            self,
            feature="ThreadGroup.rank",
            level=level,
        )
        from ._lowering import _thread_group as _provider

        return _provider.provider_group_query(
            group=group,
            op="rank",
            level=level,
            result_type=dtype,
        )

    def count_as(self, dtype: Any = None, level: str = "thread") -> Any:
        level = normalize_thread_level(
            level,
            scope=_ROOT_SCOPE,
            feature="ThreadGroup.count",
        )
        group = _validate_query_launch(
            self,
            feature="ThreadGroup.count",
            level=level,
        )
        from ._lowering import _thread_group as _provider

        return _provider.provider_group_query(
            group=group,
            op="count",
            level=level,
            result_type=dtype,
        )

    def sync(self) -> None:
        group = _validate_sync_launch(self, feature="ThreadGroup.sync")
        from ._lowering import _thread_group as _provider

        _provider.provider_group_sync(group=group, aligned=False)

    def sync_aligned(self) -> None:
        group = _validate_sync_launch(self, feature="ThreadGroup.sync_aligned")
        from ._lowering import _thread_group as _provider

        _provider.provider_group_sync(group=group, aligned=True)

    def group_by(
        self,
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup:
        return super().group_by(count, exhaustive=exhaustive)

    def is_member(self) -> Any:
        """Return whether the current thread belongs to this group."""

        group = _validate_membership_launch(
            self,
            feature="ThreadGroup.is_member",
        )
        from ._lowering import _thread_group as _provider

        return _provider.provider_group_membership(group=group)


def _require_complete_warp_partition(
    group: ThreadGroup,
    *,
    feature: str,
    exact_block_dim: tuple[int, int, int] | None = None,
) -> None:
    """Reject warp collectives whose enclosing CTA may have a partial warp."""

    if group.kind not in COMPLETE_WARP_GROUP_KINDS:
        return
    assert group.hierarchy is not None
    block_threads = group.hierarchy.block_thread_count
    if exact_block_dim is not None:
        exact_threads = 1
        for dim in exact_block_dim:
            exact_threads *= dim
        block_threads = exact_threads
    if block_threads is None:
        raise NotImplementedError(
            f"{_ROOT_SCOPE}.{feature} requires exact enclosing block dimensions "
            "to prove complete 32-thread physical-warp participation"
        )
    if block_threads % 32 != 0:
        raise NotImplementedError(
            f"{_ROOT_SCOPE}.{feature} requires every physical warp in the "
            f"enclosing CTA to be complete; got {block_threads} block threads"
        )


def _require_resolved_group(
    group: ThreadGroup,
    *,
    feature: str,
    through_level: str | None = None,
    allow_unresolved_current: bool = False,
) -> tuple[ThreadGroup, Any]:
    """Resolve one group from the active CUTLASS launch facts."""

    from ._compiler._launch import current_kernel_launch_facts

    launch = current_kernel_launch_facts()
    resolution = resolve_thread_group(
        group,
        launch,
        through_level=through_level,
    )
    if (
        resolution.unsupported is not None
        and allow_unresolved_current
        and group.is_current
        and group.mapping is None
    ):
        return group, launch
    try:
        resolved = resolution.require_supported()
    except NotImplementedError as exc:
        raise NotImplementedError(f"{_ROOT_SCOPE}.{feature} {exc}") from exc
    return resolved, launch


def _collective_group_resolution_error(
    group: ThreadGroup,
    *,
    feature: str,
) -> str:
    if group.is_static:
        return (
            f"{_ROOT_SCOPE}.{feature} could not prove that the resolved group "
            "hierarchy matches the exact kernel launch; requires exact block "
            "dimensions from verified compiler launch facts"
        )
    return (
        f"{_ROOT_SCOPE}.{feature} could not infer static block dimensions from "
        "verified compiler launch facts; requires exact block dimensions; "
        "attach reqntid to the kernel"
    )


def _resolve_collective_group_from_launch(
    group: ThreadGroup,
    launch: LaunchFacts,
    *,
    feature: str,
) -> ThreadGroup:
    """Resolve a collective group from launch facts established by its frontend."""

    resolution = resolve_thread_group(group, launch)
    if (
        resolution.unsupported is not None
        and resolution.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    ):
        raise NotImplementedError(
            _collective_group_resolution_error(group, feature=feature)
        )
    try:
        resolved = resolution.require_supported()
    except NotImplementedError as exc:
        raise NotImplementedError(f"{_ROOT_SCOPE}.{feature} {exc}") from exc
    assert resolved.hierarchy is not None
    return resolved.with_hierarchy(
        resolved.hierarchy,
        source="validated_launch" if group.is_static else "inferred_launch",
    )


def _validate_sync_launch(group: ThreadGroup, *, feature: str) -> ThreadGroup:
    """Resolve and validate the launch capabilities needed for synchronization."""

    if group.kind == "grid" and group.source == "common_root":
        raise NotImplementedError(
            f"cuda.coop.{feature} grid synchronization is unavailable through "
            "the portable API; use cuda.coop.cutlass.this_grid() under a "
            "verified cooperative launch"
        )
    if group.hierarchy.block_thread_count is not None:
        _require_complete_warp_partition(group, feature=feature)
    resolved, launch = _require_resolved_group(
        group,
        feature=feature,
        allow_unresolved_current=group.kind in {"thread", "block"},
    )
    if resolved.kind == "grid" and (
        launch.cooperative_launch is not True
        or not launch.is_verified("cooperative_launch")
    ):
        raise NotImplementedError(
            f"{_ROOT_SCOPE}.{feature} grid synchronization requires verified "
            "cooperative launch capability"
        )
    return resolved


def _validate_query_launch(
    group: ThreadGroup,
    *,
    feature: str,
    level: str,
) -> ThreadGroup:
    """Resolve source and enclosing target hierarchy for a rank/count query."""

    resolved, _ = _require_resolved_group(
        group,
        feature=feature,
        through_level=level,
        allow_unresolved_current=(
            group.kind in {"thread", "block"} and level in {"thread", "block"}
        ),
    )
    return resolved


def _validate_membership_launch(group: ThreadGroup, *, feature: str) -> ThreadGroup:
    """Resolve the group before materializing a membership predicate."""

    resolved, _ = _require_resolved_group(
        group,
        feature=feature,
        allow_unresolved_current=group.kind in PHYSICAL_GROUP_KINDS,
    )
    return resolved


def _make_group(kind: str) -> ThreadGroup:
    return make_thread_group(
        kind,
        group_type=ThreadGroup,
        scope=_ROOT_SCOPE,
    )


def this_thread() -> ThreadGroup:
    """Describe the current thread."""

    return _make_group("thread")


def this_warp() -> ThreadGroup:
    """Describe the current physical warp."""

    return _make_group("warp")


def this_block() -> ThreadGroup:
    """Describe the current CTA."""

    return _make_group("block")


def this_cluster() -> ThreadGroup:
    """Describe the current cluster where the launch can represent it."""

    return _make_group("cluster")


def this_grid() -> ThreadGroup:
    """Describe the current grid where the launch can represent it."""

    return _make_group("grid")


__all__ = [
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "cpp_level_expr",
    "render_group_decl",
    "render_group_decl_lines",
    "render_hierarchy_decl",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]

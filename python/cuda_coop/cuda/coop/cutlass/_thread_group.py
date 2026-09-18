# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Construct symbolic groups and resolve their CuTe launch dimensions.

``this_block()`` and ``this_warp()`` record the requested group without
querying a device. Primitive lowering resolves the group through the shared
resolver, adds CUTLASS operation context to failures, and checks complete
physical-warp membership in the launch.
"""

from __future__ import annotations

from cuda.coop._core import (
    COMPLETE_WARP_GROUP_KINDS,
    LaunchFacts,
    ThreadHierarchy,
    make_thread_group,
    resolve_thread_group,
)
from cuda.coop._core.thread_group import ThreadGroup as CommonThreadGroup

Hierarchy = ThreadHierarchy


class ThreadGroup(CommonThreadGroup):
    """A shared thread-group descriptor consumed by CUTLASS primitives."""

    def rank(self, level="thread"):
        """Return this group's rank relative to a hierarchy level."""
        return self.rank_as(None, level)

    def count(self, level="thread"):
        """Return this group's count relative to a hierarchy level."""
        return self.count_as(None, level)

    def rank_as(self, dtype=None, level="thread"):
        from ._lowering._thread_group import provider_group_query

        return provider_group_query(
            group=self, op="rank", level=level, result_type=dtype
        )

    def count_as(self, dtype=None, level="thread"):
        from ._lowering._thread_group import provider_group_query

        return provider_group_query(
            group=self, op="count", level=level, result_type=dtype
        )

    def sync(self):
        """Synchronize the participating members of this group."""
        from ._lowering._thread_group import provider_group_sync

        provider_group_sync(group=self, aligned=False)

    def sync_aligned(self):
        """Synchronize a converged group with aligned participation."""
        from ._lowering._thread_group import provider_group_sync

        provider_group_sync(group=self, aligned=True)

    def is_member(self):
        """Return whether this thread is included in the group."""
        from ._lowering._thread_group import provider_group_membership

        return provider_group_membership(group=self)


def _resolve_primitive_group_from_launch(
    group: ThreadGroup,
    launch: LaunchFacts,
    *,
    feature: str,
) -> ThreadGroup:
    """Resolve a primitive group using the compiler's exact launch facts.

    Keep shared resolution failures but add the qualified operation context.
    Record whether the supplied group was already static or inferred from the
    launch; constructing this descriptor does not emit a collective.
    """

    resolution = resolve_thread_group(group, launch)
    try:
        resolved = resolution.require_supported()
    except NotImplementedError as exc:
        raise NotImplementedError(f"cuda.coop.cutlass.{feature} {exc}") from exc
    assert resolved.hierarchy is not None
    return resolved.with_hierarchy(
        resolved.hierarchy,
        source="validated_launch" if group.is_static else "inferred_launch",
    )


def _require_complete_warp_partition(
    group: ThreadGroup,
    *,
    feature: str,
    exact_block_dim: tuple[int, int, int] | None = None,
) -> None:
    """Require complete 32-thread membership for a supplied warp group.

    Use an exact block-shape override when supplied, otherwise the resolved
    hierarchy. This helper does not add warp support to a primitive.
    """

    if group.kind not in COMPLETE_WARP_GROUP_KINDS:
        return
    assert group.hierarchy is not None
    block_threads = group.hierarchy.block_thread_count
    if exact_block_dim is not None:
        x, y, z = exact_block_dim
        block_threads = x * y * z
    if block_threads is None:
        raise NotImplementedError(
            f"cuda.coop.cutlass.{feature} requires exact enclosing "
            "block dimensions to prove complete 32-thread physical-warp "
            "participation"
        )
    if block_threads % 32:
        raise NotImplementedError(
            f"cuda.coop.cutlass.{feature} requires every physical warp in the "
            f"enclosing CTA to be complete; got {block_threads} block threads"
        )


def this_block() -> ThreadGroup:
    """Describe the current CUDA thread block for a CUTLASS primitive."""

    return make_thread_group(
        "block", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_warp() -> ThreadGroup:
    """Describe the calling complete 32-thread physical warp.

    A primitive resolves this symbolic group from the enclosing launch. The
    block must contain only complete physical warps, and all 32 lanes of each
    participating warp must reach the call.
    """

    return make_thread_group(
        "warp", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_thread() -> ThreadGroup:
    """Describe the calling CUDA thread."""
    return make_thread_group(
        "thread", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_cluster() -> ThreadGroup:
    """Describe the current cluster using verified launch facts."""
    return make_thread_group(
        "cluster", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_grid() -> ThreadGroup:
    """Describe the current grid using exact launch facts."""
    return make_thread_group(
        "grid", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


__all__ = [
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "_require_complete_warp_partition",
    "_resolve_primitive_group_from_launch",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]

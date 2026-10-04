# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe groups and lower their methods during CuTe tracing.

Factories describe the calling thread, warp, block, cluster, or grid.
Construction records intent without querying a device or synchronizing
threads. Group methods emit device queries or barriers during tracing.
Queries and primitives resolve the launch dimensions that each operation needs.
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
    """Describe a group and emit CuTe hierarchy queries on demand.

    Construction and group_by create descriptors. Rank, count,
    membership, and synchronization methods emit device operations when
    called during tracing. Query semantics follow cuda.coop.ThreadGroup;
    results are CuTe scalar values.
    """

    def rank(self, level="thread"):
        """Query rank with the default unsigned CuTe result type.

        See :meth:`cuda.coop.ThreadGroup.rank` for level semantics. Queries
        involving the grid return Uint64; other queries return Uint32.
        """
        return self.rank_as(None, level)

    def count(self, level="thread"):
        """Query count with the default unsigned CuTe result type.

        See :meth:`cuda.coop.ThreadGroup.count` for level semantics. Queries
        involving the grid return Uint64; other queries return Uint32.
        """
        return self.count_as(None, level)

    def rank_as(self, dtype=None, level="thread"):
        """Query rank using an optional integer dtype selector.

        The result remains a CuTe scalar even when dtype is a Python or NumPy
        type. None keeps the default unsigned width; shared group-query rules
        determine which hierarchy levels are accessible.
        """

        from ._lowering._thread_group import provider_group_query

        return provider_group_query(
            group=self, op="rank", level=level, result_type=dtype
        )

    def count_as(self, dtype=None, level="thread"):
        """Query count using an optional integer dtype selector.

        The result remains a CuTe scalar even when dtype is a Python or NumPy
        type. None keeps the default unsigned width and the query preserves
        the common level semantics.
        """

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
        """Return a CuTe Uint8 flag for membership in this group.

        Check membership before using a mapped rank. Primitive
        participation rules still apply; a membership guard alone cannot
        make a divergent collective valid.
        """
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
    """Require complete physical warps in the block for warp or mapped groups.

    Use ``exact_block_dim`` when given, otherwise the group's hierarchy.
    Raise ``NotImplementedError`` when neither gives a block size or the
    size is not a multiple of 32. This check does not decide whether the
    primitive supports the group.
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
    """Describe the calling physical 32-thread warp.

    Construction does not inspect active lanes or synchronize threads. A
    consuming query or primitive resolves the required launch facts and
    applies that operation's participation rules.
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
    """Describe the cluster of the current kernel launch.

    The consuming query or primitive resolves launch facts. Constructing this
    descriptor does not configure cluster scheduling.
    """
    return make_thread_group(
        "cluster", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_grid() -> ThreadGroup:
    """Describe the current grid for supported hierarchy queries.

    Queries resolve the launch dimensions when consumed. This descriptor does
    not enable grid synchronization or reductions.
    """
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

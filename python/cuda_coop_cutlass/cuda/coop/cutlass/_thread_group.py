# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A thread group names the threads that make a call together.

Every primitive takes a group as its first argument. ``this_block()`` is the
calling thread's block, ``this_warp()`` is its warp of 32 threads, and
``this_warp().group_by(8)`` is its group of eight neighboring lanes in that
warp. This kernel sums the same values over each of the three:

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py>`__

.. literalinclude::
    ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
    :language: python
    :start-after: # cutlass-concept-groups-begin
    :end-before: # cutlass-concept-groups-end

.. figure:: visualizations/concept-thread-groups.svg
   :alt: Five boxes from grid to thread, each named by its this_ function,
         with block and warp filled. Below them, one block of 128 threads
         is drawn as one group, as four warps, and as sixteen groups of
         eight lanes. A dot marks rank 0 of each group.
   :width: 100%

   The three groups of the kernel above, on its block of 128 threads.

The group selects the CUB class that runs. Six families work on a block or
a warp, one on warps only, and the rest on blocks only:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Primitive
     - ``this_block()``
     - ``this_warp()``, ``this_warp().group_by(n)``
   * - ``load``, ``store``
     - ``cub::BlockLoad``, ``cub::BlockStore``
     - ``cub::WarpLoad``, ``cub::WarpStore``
   * - ``reduce``, ``sum``
     - ``cub::BlockReduce``
     - ``cub::WarpReduce``
   * - ``scan`` and its four helpers
     - ``cub::BlockScan``
     - ``cub::WarpScan``
   * - ``exchange``
     - ``cub::BlockExchange``
     - ``cub::WarpExchange``
   * - ``merge_sort_keys``, ``merge_sort_pairs``
     - ``cub::BlockMergeSort``
     - ``cub::WarpMergeSort``
   * - ``reduce_batched``
     -
     - ``cub::WarpReduceBatched``
   * - ``shuffle``
     - ``cub::BlockShuffle``
     -
   * - ``radix_sort_keys``, ``radix_sort_pairs``, ``radix_rank_keys``
     - ``cub::BlockRadixSort``, ``cub::BlockRadixRank``
     -
   * - The four ``topk`` functions
     - CUB's block top-k
     -
   * - ``adjacent_difference``, ``discontinuity``
     - ``cub::BlockAdjacentDifference``, ``cub::BlockDiscontinuity``
     -
   * - ``histogram``
     - ``cub::BlockHistogram``
     -
   * - ``run_length_decode``, ``run_length_decode_into``
     - ``cub::BlockRunLengthDecode``
     -

These rules hold for every primitive:

* Every thread of the group makes the call, including the threads that do
  not use its result. A warp call involves that warp only, so another warp
  of the block can skip it.
* The launch states the block size with constants, as in
  ``launch(grid=1, block=128)``. Each call is compiled for that size, and a
  kernel whose block size is known only at run time is rejected. Most
  primitives take blocks of one, two or three dimensions. Threads are
  ranked x first: ``x + block_x * (y + block_y * z)``.
* A warp group needs a block whose size is a multiple of 32, so that every
  warp is complete.
* Each block works on its own tile. A kernel with more than one block gives
  Load and Store an ``offset`` computed from ``cute.arch.block_idx()``.

``this_thread()``, ``this_cluster()`` and ``this_grid()`` name the other
levels of a launch, and ``this_block().group_by(n)`` names ``n`` whole warps.
These four are for questions and barriers. Any group can tell a thread where
it is:

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py>`__

.. literalinclude::
    ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
    :language: python
    :start-after: # cutlass-concept-group-queries-begin
    :end-before: # cutlass-concept-group-queries-end
    :dedent: 4

``rank`` and ``count`` take a level: ``"thread"`` (the default), ``"warp"``,
``"block"``, ``"cluster"`` or ``"grid"``. For a level inside the group,
``rank`` is where this thread, or its warp, sits in the group, and ``count``
is how many the group holds. For a level outside the group, ``rank`` is
where the group sits in that level, and ``count`` is how many such groups
the level holds. The results are CuTe scalars, so a kernel can use them in
arithmetic, as an index, or in a condition. A query emits no barrier and
uses no shared memory, and only the threads that want the answer need to
ask.

``sync()`` is a barrier for the threads of the group:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Group
     - ``sync()``
   * - ``this_block()``
     - Block barrier
   * - ``this_warp()``
     - Warp barrier, ``__syncwarp()``
   * - ``this_warp().group_by(n)``
     - Warp barrier for the lanes of that group
   * - ``this_cluster()``
     - Cluster barrier on targets that have one, from ``sm_90``. Block
       barrier on older targets.
   * - ``this_thread()``
     - Nothing

``this_grid()`` and ``this_block().group_by(n)`` have no barrier.
"""

# Implementation notes.
#
# Factories describe the calling thread, warp, block, cluster, or grid.
# Construction records intent without querying a device or synchronizing
# threads. Group methods emit device queries or barriers during tracing.
# Queries and primitives resolve the launch dimensions that each operation
# needs.

from __future__ import annotations

from cuda.coop.cutlass._planning import (
    COMPLETE_WARP_GROUP_KINDS,
    LaunchFacts,
    ThreadHierarchy,
    make_thread_group,
    resolve_thread_group,
)
from cuda.coop.cutlass._planning.thread_group import (
    ThreadGroup as PlanningThreadGroup,
)

Hierarchy = ThreadHierarchy


class ThreadGroup(PlanningThreadGroup):
    """The threads that make a call together.

    A kernel gets one from :func:`cuda.coop.cutlass.this_block`,
    :func:`cuda.coop.cutlass.this_warp` or another ``this_`` function, and
    splits a warp or a block with
    :meth:`~cuda.coop.cutlass.ThreadGroup.group_by`. Creating a group emits
    no code. A primitive or one of the methods below does.
    """

    def rank(self, level="thread"):
        """Return where this thread, or this group, sits.

        Parameters
        ----------
        level : str, optional
            ``"thread"`` (the default), ``"warp"``, ``"block"``,
            ``"cluster"`` or ``"grid"``. ``"gpu_thread"`` also means
            ``"thread"``.

        Returns
        -------
        cutlass.Uint32 or cutlass.Uint64
            For a level inside the group, the position within the group of
            this thread's unit of that level. ``this_block().rank()`` is the
            thread's position in its block, and
            ``this_block().rank("warp")`` is its warp's. For a level outside
            the group, the group's position in that level:
            ``this_block().rank("grid")`` is the block's position in the
            grid. The result is ``Uint64`` when the group or the level is
            the grid.

        Notes
        -----
        In a block with more than one dimension, threads are ranked x first:
        ``x + block_x * (y + block_y * z)``.
        """
        return self.rank_as(None, level)

    def count(self, level="thread"):
        """Return the size of the group, or how many such groups there are.

        Parameters
        ----------
        level : str, optional
            A level name, as for
            :meth:`~cuda.coop.cutlass.ThreadGroup.rank`.

        Returns
        -------
        cutlass.Uint32 or cutlass.Uint64
            For a level inside the group, the number of those units in the
            group: ``this_block().count()`` is the block's thread count and
            ``this_block().count("warp")`` is its warp count. For a level
            outside the group, the number of such groups in that level:
            ``this_block().count("grid")`` is the number of blocks in the
            grid. The result is ``Uint64`` when the group or the level is
            the grid.
        """
        return self.count_as(None, level)

    def rank_as(self, dtype=None, level="thread"):
        """Return the rank as the integer type you name.

        Parameters
        ----------
        dtype : type, optional
            A signed or unsigned integer type of 8, 16, 32 or 64 bits: a
            CUTLASS type such as ``cutlass.Int32``, a NumPy integer type, or
            Python ``int``, which means ``Int32``. ``None`` keeps the
            default of :meth:`~cuda.coop.cutlass.ThreadGroup.rank`.
        level : str, optional
            A level name, as for
            :meth:`~cuda.coop.cutlass.ThreadGroup.rank`.

        Returns
        -------
        CuTe integer scalar
            The rank, cast to the requested type without a range check.
        """

        from ._lowering._thread_group import provider_group_query

        return provider_group_query(
            group=self, op="rank", level=level, result_type=dtype
        )

    def count_as(self, dtype=None, level="thread"):
        """Return the count as the integer type you name.

        Parameters
        ----------
        dtype : type, optional
            An integer type, as for
            :meth:`~cuda.coop.cutlass.ThreadGroup.rank_as`.
        level : str, optional
            A level name, as for
            :meth:`~cuda.coop.cutlass.ThreadGroup.rank`.

        Returns
        -------
        CuTe integer scalar
            The count, cast to the requested type without a range check.
        """

        from ._lowering._thread_group import provider_group_query

        return provider_group_query(
            group=self, op="count", level=level, result_type=dtype
        )

    def sync(self):
        """Wait until every thread of the group arrives.

        A block waits at a block barrier and a warp at ``__syncwarp()``. A
        group of lanes waits for its own lanes only. A cluster waits at a
        cluster barrier on targets from ``sm_90``, and at a block barrier
        on older ones. ``this_thread().sync()`` emits nothing. The grid and
        groups of warps have no barrier, and calling ``sync`` on them is
        rejected when the kernel is compiled.
        """
        from ._lowering._thread_group import provider_group_sync

        provider_group_sync(group=self, aligned=False)

    def sync_aligned(self):
        """Wait at the aligned form of the group's barrier.

        For a block this is ``__syncthreads()``, and a cluster has an
        aligned barrier as well. Use it only where every thread of the
        group executes this same call. In a branch whose condition differs
        between threads, use :meth:`~cuda.coop.cutlass.ThreadGroup.sync`.
        For a warp or a group of lanes the two methods are the same.
        """
        from ._lowering._thread_group import provider_group_sync

        provider_group_sync(group=self, aligned=True)

    def is_member(self):
        """Return whether the calling thread belongs to this group.

        Returns
        -------
        cutlass.Uint8
            One for every thread, except in a group made with
            ``group_by(count, exhaustive=False)``, where the lanes or warps
            left over get zero.

        Notes
        -----
        A thread that is not a member skips the group's primitives, and its
        ``rank`` in the group means nothing.
        :meth:`~cuda.coop.cutlass.ThreadGroup.group_by` has an example.
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

    Keep shared resolution failures but add the operation context.
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
    """Return the calling thread's block.

    Block primitives take it as their first argument. Its size comes from
    the ``block`` argument of the launch.
    """

    return make_thread_group(
        "block", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_warp() -> ThreadGroup:
    """Return the calling thread's warp of 32 threads.

    Warp primitives take it as their first argument. The block size must be
    a multiple of 32, so that every warp is complete.
    :meth:`ThreadGroup.group_by <cuda.coop.cutlass.ThreadGroup.group_by>`
    splits it into smaller groups.
    """

    return make_thread_group(
        "warp", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_thread() -> ThreadGroup:
    """Return the calling thread as a group of one, for queries."""
    return make_thread_group(
        "thread", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_cluster() -> ThreadGroup:
    """Return the calling thread's cluster of blocks, for queries and ``sync``.

    The ``cluster`` argument of the launch sets how many blocks a cluster
    has. Without it, each block is a cluster by itself.
    """
    return make_thread_group(
        "cluster", group_type=ThreadGroup, scope="cuda.coop.cutlass"
    )


def this_grid() -> ThreadGroup:
    """Return every thread of the launch as one group, for queries.

    Its ``rank`` and ``count`` return ``Uint64``. The launch states the grid
    size with constants, as it does the block size. The grid has no barrier.
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

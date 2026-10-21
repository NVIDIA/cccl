.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Hierarchy and group queries
===========================

``cuda-coop`` describes the current CUDA launch without requiring dimensions in
the Rust source. ``this_thread()``, ``this_warp()``, ``this_block()``,
``this_cluster()`` and ``this_grid()`` return typed group descriptors.
``ThreadHierarchy::new()`` and ``ThreadHierarchy::current()`` provide the same
physical groups through ``thread()``, ``warp()``, ``block()``, ``cluster()`` and
``grid()``. ``Hierarchy`` aliases ``ThreadHierarchy``.

Queries run in an Oxide-compiled kernel with the workspace's pinned
``nightly-2026-10-11`` toolchain. The device dependency is pinned in Cargo; there
is no backend-selection feature. Host payload tests require no CUDA installation.
The host provider remains independently usable on Rust 1.85.

Rank and count
--------------

``group.rank()`` and ``group.count()`` default to the Thread level. Rust selects
an explicit level with ``rank_at::<coop::Warp>()`` or
``count_at::<coop::Warp>()``. An inner level queries constituents within the
group; an outer level queries that group within its parent. Equal levels have
rank zero and count one. Physical queries use x-fastest linearization within
each level, with blocks ordered before their constituent threads or warps.

For example, ``this_block().rank()`` is the current thread's linear rank in the
block, while ``this_block().rank_at::<coop::Grid>()`` is the block's linear rank
in the grid. The result is ``u64`` when either scope or query level is Grid,
and ``u32`` otherwise. Grid arithmetic promotes coordinates before multiplying.
``rank_as::<i16>()`` and ``count_as::<i16>()`` select one of the eight fixed-width
integer types; ``rank_at_as::<i16, coop::Warp>()`` also selects a level.
Conversions follow integer ``as`` semantics, so callers must choose a sufficient
width to avoid truncation.

A physical warp always counts 32 lanes. Block warp counts round up when the
last warp is partially occupied. On known targets below sm_90, cluster queries
describe one block per cluster.
On sm_90 and newer they describe the launch's cluster; an ordinary, unclustered
launch still has one block per cluster. An unknown target retains the sm_90
minimum for these queries. No query allocates storage or synchronizes threads.
Copying a descriptor preserves the
scope, and subsequent queries describe the thread executing them.

Mapped groups
-------------

``this_warp().group_by::<8>()`` partitions a physical warp into groups of eight
lanes. ``this_block().group_by::<2>()`` partitions a block into groups of two
**physical warps**, each containing 64 threads. The default is exhaustive:
partition size must divide the parent's unit count. Both forms require complete
physical warps in the enclosing block.

``group_by_non_exhaustive::<N>()`` excludes leftover units. Its ``is_member()``
query returns ``1u8`` for members and ``0u8`` for excluded threads. Counts remain
defined for nonmembers; rank queries on a nonmember trap. Lane groups accept
only Thread and Warp levels; warp groups accept Thread, Warp and Block levels.
Mapped groups cannot be partitioned again.

Sizes must be positive compile-time constants. Invalid static sizes fail during
compilation. Checks that depend on the actual block dimensions trap on device
when violated. These descriptors and checks provide no proof of collective
participation, uniform control flow, or scratch ownership. The Load/Store API
accepts power-of-two lane groups of widths 1, 2, 4, 8, 16 and 32 with Direct and
implicit storage. Other mapped groups remain query-only. Canonical Warp-in-Block
and logical-group-in-Warp ranks preserve group uniformity in checked operations;
ordinary lane arithmetic does not receive that guarantee.

Run the native query checks
---------------------------

The standalone ``rust/cuda-coop/tests/device_queries`` package contains the
complete native program. It verifies all physical scope/level pairs with
multidimensional blocks and grids, a partial final warp, a clustered launch,
exhaustive and non-exhaustive mappings, and integer conversions. Raw CUDA
coordinates select output locations independently of the queries being checked.
The same program checks payload initialization, safe access and copying at
item counts 1 and 4. It uses the
same pinned public Oxide dependencies as the library. Keeping this native
package separate lets ordinary workspace tests run without CUDA host libraries.

With the matching ``cargo-oxide`` compiler installed, a CUDA toolkit, and an
sm_120 GPU, run from the CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/tests/device_queries
   CUDA_HOME=/usr/local/cuda-13.3 cargo oxide run --arch sm_120 --materialize-cubin
   compute-sanitizer --tool memcheck --error-exitcode 1 target/release/device_queries

The five rejection modes run in separate processes because a device trap makes
its CUDA context unusable:

.. code-block:: shell

   for case in 0 1 2 3 4; do
     target/release/device_queries "$case"
   done

They check partial physical warps, an indivisible exhaustive partition, a
partition larger than its parent, and a rank query by an excluded lane. The
last mode indexes an unwritten payload slot. The program reports success only
when CUDA returns the expected device-trap error.

.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Hierarchy and group queries
==========================

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
rank zero and count one. Physical queries use x-fastest linearization.

For example, ``this_block().rank()`` is the current thread's linear rank in the
block, while ``this_block().rank_at::<coop::Grid>()`` is the block's linear rank
in the grid. The result is ``u64`` when either scope or query level is Grid,
and ``u32`` otherwise. Grid arithmetic promotes coordinates before multiplying.
``rank_as::<i16>()`` and ``count_as::<i16>()`` select one of the eight fixed-width
integer types; ``rank_at_as::<i16, coop::Warp>()`` also selects a level.
Conversions follow integer ``as`` semantics, so callers must choose a sufficient
width to avoid truncation.

A physical warp always counts 32 lanes. Block warp counts round up when the
last warp is partially occupied. Cluster queries currently require sm_90 or
newer; an ordinary, unclustered launch has one block per cluster. No query
allocates storage or synchronizes threads. Copying a descriptor preserves the
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
participation, uniform control flow, or scratch ownership. Synchronization,
temporary storage and collective primitives are separate work.

Run the native query checks
---------------------------

The standalone ``rust/cuda-coop/tests/device_queries`` package contains the
complete native program. It verifies all physical scope/level pairs with
multidimensional blocks and grids, a partial final warp, a clustered launch,
exhaustive and non-exhaustive mappings, and integer conversions. It uses the
same pinned public Oxide dependencies as the library. Keeping this native
package separate lets ordinary workspace tests run without CUDA host libraries.

With the matching ``cargo-oxide`` compiler installed, a CUDA toolkit, and an
sm_120 GPU, run from the CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/tests/device_queries
   CUDA_HOME=/usr/local/cuda-13.3 cargo oxide run --arch sm_120 --materialize-cubin
   compute-sanitizer --tool memcheck --error-exitcode 1 target/release/device_queries

The four rejection modes run in separate processes because a device trap makes
its CUDA context unusable:

.. code-block:: shell

   for case in 0 1 2 3; do
     target/release/device_queries "$case"
   done

They check partial physical warps, an indivisible exhaustive partition, a
partition larger than its parent, and a rank query by an excluded lane. The
program reports success only when CUDA returns the expected device-trap error.

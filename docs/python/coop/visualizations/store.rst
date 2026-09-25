.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-store:

Store
=====

:func:`cuda.coop.store` writes each thread's values into a contiguous memory
tile. The algorithm determines which input ownership it expects and whether
a shared-memory exchange rearranges the values before the writes.

The explorer shows eight illustrative threads and full-tile stores. The
warp-transpose variants use two four-lane teaching warps; **CUDA warps have
32 lanes**. Executable blocks using those variants must contain a multiple
of 32 threads.

.. coop-visualization:: store

   .. only:: html

      .. figure:: store-transpose.svg
         :alt: Transpose store exchanges blocked pairs into striped writer registers, then stores the values in consecutive memory locations.
         :width: 100%

         Transpose store with eight illustrative threads and two items per thread.

   Each thread begins with a consecutive pair. The exchange redistributes
   those values so adjacent threads can write adjacent memory locations.

   .. code-block:: text

      Thread          0     1     2     3     4     5     6     7
      Input          0,1   2,3   4,5   6,7   8,9  10,11 12,13 14,15
      Writer values  0,8   1,9  2,10  3,11  4,12  5,13  6,14  7,15
      Memory         0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

Reading the stages
------------------

See :ref:`blocked versus striped <coop-glossary-layouts>` for a compact
ownership table and the index formulas used here.

The first row shows a working copy of the input payload. Store preserves
the caller's payload, including when an exchange rearranges its internal
copy. For algorithms with an exchange, values pass through shared scratch
before reaching the writer-register row. The final row shows memory
positions. Select a value to follow its input thread, writer thread, and
destination address.

``direct`` and ``vectorize`` consume blocked input: thread ``t`` provides
``K`` consecutive values beginning at logical position ``t * K``.
Vectorization can combine adjacent values into wider stores when the
pointer, alignment, type, and item count allow it. The illustration does
not guarantee a particular instruction or memory-transaction count.

``striped`` consumes striped input: thread ``t`` writes slot ``j`` to
``t + j * T``, where ``T`` is the group size. The input must already have
that ownership. ``transpose`` instead accepts blocked input and exchanges
it into striped writer registers before performing the same writes.

``warp_transpose`` performs the corresponding exchange within each physical
warp. ``warp_transpose_timesliced`` reuses one warp's shared scratch across
serialized exchange rounds. The writer registers then feed warp-striped
stores; the diagram does not impose a serial global-memory store schedule.

Using Store in a kernel
-----------------------

This common-API fragment works in either DSL with the
:ref:`kernel-fragment setup <coop-visualization-kernels>`. Launch with
128 threads and provide at least 256
source and destination elements for each block.

.. code-block:: python

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   offset = block_index * 256
   coop.load(block, source, items, algorithm="direct", offset=offset)
   coop.store(block, destination, items, algorithm="transpose", offset=offset)
   # The block's 256 values are written in their original logical order.
   # Store writes memory and returns None.

For a partial final tile, pass ``valid_items`` to limit the stored prefix;
memory beyond that prefix remains untouched. Supply a matching valid count
when loading the input tile. See :func:`cuda.coop.store` for the parameter
contract and the :doc:`Numba <../programming_guide>` and
:ref:`CUTLASS <coop-cutlass-load-store>` Load/Store examples for launches
and partial tiles.

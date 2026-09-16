.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-exchange:

Exchange
========

:func:`cuda.coop.exchange` rearranges register values across a group. It
returns a new payload with the requested ownership and preserves the input
payload. Exchange itself performs no global-memory load or store.

The common API provides ``striped_to_blocked`` and ``blocked_to_striped``.
The explorer also includes the six additional block modes available through
``cuda.coop.numba_mlir.exchange``: two warp-striped conversions and four
rank-based scatters. Each option identifies which API provides it.

.. coop-visualization:: exchange

   .. only:: html

      .. figure:: exchange-striped.svg
         :alt: Striped-to-blocked exchange turns thread-zero values zero and eight into zero and one, with corresponding consecutive pairs for the other threads.
         :width: 100%

         Striped-to-blocked exchange with eight illustrative threads and two items per thread.

   Values retain their logical sequence while their owning threads change.

   .. code-block:: text

      Thread          0     1     2     3     4     5     6     7
      Striped input  0,8   1,9  2,10  3,11  4,12  5,13  6,14  7,15
      Blocked result 0,1   2,3   4,5   6,7   8,9  10,11 12,13 14,15

Reading the stages
------------------

The first row groups inputs by thread. The middle row shows shared scratch
at logical positions; the final row groups the returned values by their
new owners. Select a value to inspect its input and result slots.

For ``K`` items per thread and ``T`` threads, blocked ownership places
logical value ``p`` in thread ``p // K``, slot ``p % K``. Striped ownership
places it in thread ``p % T``, slot ``p // T``. The two common modes convert
between these arrangements without changing the logical order.

``blocked_to_warp_striped`` and ``warp_striped_to_blocked`` apply that
conversion separately within each physical warp's tile. The explorer uses
four lanes per teaching warp to fit two warp tiles on screen. **Physical
CUDA warps have 32 lanes.** Warp-striped ownership differs from stripes
across a block containing multiple warps.

The scatter modes take a signed-integer ``ranks`` payload with the same
item count as the input. Each rank selects a logical result position;
``scatter_to_blocked`` and ``scatter_to_striped`` expose those destinations
in their named layouts. The explorer uses ``rank[p] = (5 * p) % N``, a
permutation for the illustrated power-of-two tile sizes. Active ranks must
be in range and unique.

``scatter_to_striped_guarded`` suppresses inputs with negative ranks; it
does not guard against ranks beyond the tile extent. The example assigns
rank ``-1`` to every fifth input. ``scatter_to_striped_flagged`` instead
uses an integral ``valid_flags`` payload and suppresses every fourth input
in the example. Suppressed writes leave holes: **a question mark means an
unspecified output that must not be consumed**, not a zero or a preserved
input value.

The illustrations use the default non-timesliced exchange. The qualified
block API also provides ``warp_time_slicing`` for the layout conversions
and unguarded scatters. Guarded and flagged scatter do not support it.

Using Exchange in a kernel
--------------------------

This fragment uses the common API inside a Numba-CUDA-MLIR kernel, with
``cuda`` imported from ``numba_cuda_mlir``, ``numpy as np``, and
``cuda.coop as coop``. Launch with 128 threads and provide at least 256
source and destination elements for each block.

.. code-block:: python

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   offset = cuda.blockIdx.x * 256
   coop.load(block, source, items, algorithm="striped", offset=offset)
   blocked = coop.exchange(block, items, mode="striped_to_blocked")
   coop.store(block, destination, blocked, algorithm="direct", offset=offset)
   # blocked owns consecutive pairs; items retains striped ownership.

To scatter, import ``cuda.coop.numba_mlir as numba_coop`` and use its
qualified operation. With the same 128-thread, two-item launch, the
following full-tile permutation writes every destination exactly once:

.. code-block:: python

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   ranks = coop.ThreadData(2, dtype=np.int32)
   offset = cuda.blockIdx.x * 256
   coop.load(block, source, items, algorithm="direct", offset=offset)
   for item in range(2):
       position = cuda.threadIdx.x * 2 + item
       ranks[item] = (5 * position) % 256
   striped = numba_coop.exchange(
       block, items, mode="scatter_to_striped", ranks=ranks
   )
   coop.store(block, destination, striped, algorithm="striped", offset=offset)

See :func:`cuda.coop.exchange` for the common operation and the
:doc:`../programming_guide` for backend activation and group participation.

.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-faqs:

FAQs
====

.. _coop-faq-namespaces:

Why are there both ``cuda.coop`` and ``cuda.coop.numba_mlir``?
--------------------------------------------------------------

``cuda.coop`` provides the common API for cooperative operations. A kernel
compiler's backend implements those calls. Start with this namespace when
its groups, ``ThreadData`` payloads, and built-in operators cover your needs:

.. code-block:: python

   from cuda import coop

   coop.register("numba-cuda-mlir")

``cuda.coop.numba_mlir`` exposes that backend's API, including extensions
specific to Numba-CUDA-MLIR. Use it for features such as fixed-size Numba
local-array payloads, device callbacks, or Scan prefix callbacks:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop

Importing this namespace also registers the backend. Both namespaces can
appear in one kernel, and shared operations follow the same contracts.
See the :ref:`API comparison <coop-programming-api-choice>` for the
operation-specific differences.

Numba-CUDA-MLIR is the first backend; CUTLASS support is planned. The common
API gives libraries a compiler-independent way to express cooperative
operations. Kernel launch syntax and other DSL-specific code still need
adaptation when moving to another compiler.

.. _coop-faq-numba-only:

I only use Numba-CUDA-MLIR. Can I import its namespace as ``coop``?
-------------------------------------------------------------------

Yes. This is supported and registers the backend:

.. code-block:: python

   from numba_cuda_mlir import cuda

   import cuda.coop.numba_mlir as coop

You can use common operations and backend extensions through that one name.
The documentation uses ``numba_coop`` when showing backend calls alongside
common calls, so readers can see which API an example needs.

Keep the alias on a dotted import. Bare ``import cuda.coop.numba_mlir``
assigns the top-level package to ``cuda`` in that scope, replacing the name
previously imported from ``numba_cuda_mlir``.

.. _coop-faq-temp-storage:

Why or when would I provide my own ``TempStorage``?
---------------------------------------------------

Usually, you can omit it. The compiler allocates the scratch required by
an operation and inserts a barrier to make reuse safe. Direct, striped, and
vectorized Load/Store need no shared scratch.

An explicit descriptor is useful when several supported block operations
can reuse the same allocation. For example, a transpose Load and Store can
share scratch while the values stay in each thread's payload:

.. code-block:: python

   # Inside a kernel; source and destination are kernel arguments.
   block = coop.this_block()
   items = coop.ThreadData(2)
   scratch = coop.TempStorage()
   coop.load(
       block, source, items, algorithm="transpose", temp_storage=scratch
   )
   coop.store(
       block, destination, items, algorithm="transpose", temp_storage=scratch
   )

The compiler sizes and aligns the shared region for both calls. It inserts
a barrier after each use. Construct the descriptor inside the kernel, and
keep application values in ``ThreadData`` or your own arrays.

A descriptor also lets you request capacity or alignment, choose separate
slices with ``sharing="exclusive"``, or take responsibility for reuse barriers
with ``auto_sync=False``. Keep automatic synchronization enabled unless your
kernel provides the required barriers itself, including across loop
iterations. Separate slices do not remove the need to protect reuse.

The current backend accepts explicit descriptors for block transpose-family
Load/Store, Block Scan, Block Merge Sort, Block Radix Sort, TopK, Adjacent
Difference, Discontinuity, Histogram, and both Run Length Decode forms.
Warp Load/Store, Warp Scan, Warp Merge Sort, and Batched Warp Reduction use
compiler-owned storage. See
:ref:`temporary storage <coop-temp-storage>` for the complete contract and
shared-memory restrictions.

.. _coop-faq-installed-extra:

Can ``cuda.coop`` tell which extra I installed?
-----------------------------------------------

Not reliably. Package metadata lists the extras a distribution offers and
the dependencies associated with them. It does not provide a portable record
of which extra was requested when installing. The same dependencies may also
have been installed separately or by another package.

``pip install cuda-coop`` and
``pip install "cuda-coop[numba-cuda-mlir-cu13]"`` install the same wheel,
including ``cuda.coop.numba_mlir`` and every other shipped DSL integration.
The base install declares no Python package dependencies. The extra only
adds the requirements in ``pyproject.toml`` that install the supported
Numba-CUDA-MLIR stack for CUDA 13. If those dependencies are already present
at supported versions, either command gives you the same usable API.
Registration selects which compiler hooks to activate in the running process.

Use ``coop.register("numba-cuda-mlir")`` to state that intent explicitly.
It works regardless of import order, is safe to repeat, and also accepts
``"numba_cuda_mlir"``. The backend dependencies must already be installed.
See :ref:`backend registration <coop-backend-registration>`.

.. _coop-faq-topk-order:

Does TopK return sorted results?
-------------------------------

No. It selects the smallest or largest keys and places them in a blocked
output prefix without promising their order. Only the first
``min(k, valid_items)`` positions are defined, and ties at the boundary
have no ordering guarantee. Pair variants keep each selected key attached
to its value. Use a sorting primitive when you need ordered output.
See :ref:`the TopK example <coop-topk>`.

.. _coop-faq-global-sort:

Does sorting each block sort the whole array?
--------------------------------------------

Each call sorts only the selected group's tile. Several blocks therefore
produce independently sorted tiles. A globally sorted array requires an
algorithm that combines those tiles. Warp and logical-warp Merge Sort
likewise sort each participating group's tile independently.

.. _coop-faq-neighbor-operations:

When should I use Adjacent Difference or Discontinuity?
-----------------------------------------------------

Use :func:`cuda.coop.adjacent_difference` to compute a value from each item
and its neighbor, such as the delta between successive samples. Use
:func:`cuda.coop.discontinuity` to produce flags that mark boundaries, such
as the first and last item of each equal-value run. It returns ``int32``
flags; ``mode="heads_and_tails"`` returns both payloads.

Tile boundaries matter for both operations. Supply a predecessor or
successor when comparisons must continue across tiles. A multiblock
kernel that reads global neighbors must preserve those source values
until all readers finish. The :doc:`neighbor examples <neighbor-operations>`
use separate input and output arrays.

.. _coop-faq-histogram-padding:

Can I zero-pad a partial Histogram tile?
--------------------------------------

Every input sample contributes to a bin, including a padded zero. A
zero-padded load therefore adds extra counts to bin zero. Histogram has
no ``valid_items`` parameter. Process complete tiles or handle the tail
separately with a kernel that counts only valid samples.

Output padding is different: returned counter slots whose bin index is
at least ``bins`` contain zero. Store the first ``bins`` counters using
the striped layout. See the :doc:`Histogram explorer <visualizations/histogram>`.

.. _coop-faq-histogram-accumulation:

Does Histogram retain counters between calls?
--------------------------------------------

Each call returns fresh counts and preserves its samples. For repeated
tiles within a kernel, keep an accumulator payload and add the returned
counts to it. Each thread retains the same striped bin ownership when
the configuration stays fixed. The :doc:`tested accumulation example
<visualizations/histogram>` shows this pattern.

CUB's ``BlockHistogram::Composite`` accumulates into a caller's counter
buffer. That persistent state belongs to the buffer; it does not require
a persistent C++ Histogram object. The Python API exposes the fresh-count
operation, so neither a parent object nor retained counters in
``TempStorage`` are needed.

.. _coop-faq-rld-lifecycle:

Why do windowed and bulk Run Length Decode use different calls?
-------------------------------------------------------------

:func:`cuda.coop.run_length_decode` returns a fixed-size payload for the
window beginning at ``decoded_window_offset``. Use it when the kernel
needs to work with that window's values. It prepares the run table on
each call, even when calls share a ``TempStorage`` descriptor.

:func:`cuda.coop.run_length_decode_into` writes the entire expanded
sequence into a destination array. It prepares the table once, loops
over decoding windows internally, and returns the total decoded size to
every thread. The destination must have enough remaining capacity;
use separate source and destination storage.

The prepared run table persists only within the call. A window offset
selects a position in the expanded sequence; the decoder does not advance
a hidden cursor. A shared scratch descriptor controls allocation reuse,
not decoder state. See :ref:`run positions and windows
<coop-glossary-decoding>` and the
:doc:`RLD explorer <visualizations/run-length-decode>`.

.. _coop-faq-rld-padding:

How do I pad run inputs and recognize the end of decoded output?
--------------------------------------------------------------

Use a positive prefix of run lengths followed by zeros. An all-zero tile
represents an empty sequence; an interior zero followed by a positive
length is invalid. The values associated with padding runs are ignored.

A windowed decode fills positions beyond the expanded sequence with
zero. Zero may also be a real run value, so use the total decoded size
to determine which positions are valid. The qualified API can write
that total and relative run offsets to auxiliary payloads; invalid
relative offsets contain the maximum value of the selected unsigned
offset dtype. Bulk decoding writes only valid items, leaving the rest
of the destination unchanged.

.. _coop-faq-batched-reduce:

How does Batched Warp Reduction differ from ordinary Reduce?
-----------------------------------------------------------

Ordinary ``reduce(group, values)`` combines the group's payload items
into one aggregate. ``reduce_batched(warp, values)`` reduces each local
slot independently across the warp. Three slots per lane mean three
independent results, one for each slot.

The results are distributed among lanes in blocked or striped order;
they are not broadcast to every lane. Each returned payload has
``ceil(batches / warp_width)`` slots, and slots without a corresponding
batch are unspecified. The :doc:`feature-sum example
<visualizations/reduce-batched>` guards its stores by batch index.

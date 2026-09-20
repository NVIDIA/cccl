.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-glossary:

Glossary
========

Terms
-----

.. glossary::
   :sorted:

   backend
      The integration that lets a kernel compiler recognize ``cuda.coop``
      calls and generate CUDA code for them. The current integrations target
      Numba-CUDA-MLIR and CUTLASS / CuTe DSL. See :ref:`registration <coop-backend-registration>`.

   blocked
      A layout in which each thread owns consecutive elements of a
      :term:`tile`. With ``K`` items per thread, thread ``t`` owns tile
      positions ``t * K`` through ``t * K + K - 1``. See
      :ref:`blocked versus striped <coop-glossary-layouts>`.

   primitive
      .. raw:: html

         <span id="term-collective"></span>

      A cooperative operation provided by ``cuda.coop``, such as ``load``,
      ``reduce``, or ``exclusive_sum``. Each primitive defines which threads
      participate, how it uses their data, and where its results are
      available. See
      :ref:`participation and synchronization <coop-common-participation>`.

   family
      A group of related :term:`primitives <primitive>` that share semantics
      and implementation. For example, the Scan family includes ``scan``,
      ``exclusive_scan``, ``inclusive_scan``, ``exclusive_sum``, and
      ``inclusive_sum``. Families organize implementation modules; a
      :term:`thread group` describes the threads executing a primitive.
      See the implementation discussions in the
      :doc:`Numba-CUDA-MLIR <developer_overview>` and
      :doc:`CUTLASS <cutlass_developer_guide>` Developer Guides.

   payload
      The values contributed or received by one thread. ``ThreadData(K)``
      describes a fixed-size payload of ``K`` items. Some operations also
      accept scalars or backend-specific arrays. See
      :ref:`thread data <coop-common-payloads>`.

   batch
      One independent reduction in :func:`cuda.coop.reduce_batched`.
      Each lane contributes the value in the same local payload slot:
      slot ``j`` contributes to batch ``j``. A batch spans the selected
      warp, and its result belongs to the lane and slot selected by the
      output layout. See :doc:`batched reduction <visualizations/reduce-batched>`.

   bin
      A counter indexed by an input sample in :func:`cuda.coop.histogram`.
      A sample with value ``b`` increments bin ``b``. Every sample must
      satisfy ``0 <= b < bins``. Returned bins use :term:`striped`
      ownership. See :doc:`Histogram <visualizations/histogram>`.

   tile boundary
      The edge between one group's tile and the neighboring data.
      Adjacent Difference and Discontinuity can consume an explicit
      predecessor or successor value to compare across this edge.
      Without one, their endpoint rules apply to the local tile. See
      :doc:`neighbor operations <neighbor-operations>`.

   run
      A consecutive sequence represented by one value and its repetition
      count, such as value ``7`` and length ``3`` representing ``7, 7, 7``.
      Run Length Decode expands these pairs. Its run-length payload uses
      positive lengths followed by optional zero-length padding. See
      :doc:`Run Length Decode <visualizations/run-length-decode>`.

   head flag
      An integer flag marking the start of a sequence according to
      Discontinuity's comparison predicate. With the default inequality
      predicate, it is one where an item differs from its predecessor.
      Without an explicit tile predecessor, the first item is a head.
      See :doc:`Discontinuity <visualizations/discontinuity>`.

   tail flag
      An integer flag marking the end of a sequence according to
      Discontinuity's comparison predicate. With the default inequality
      predicate, it is one where an item differs from its successor.
      Without an explicit tile successor, the last item is a tail.

   decoded window
      The output interval returned by one :func:`cuda.coop.run_length_decode`
      call. Its start is an offset in the expanded sequence, and its
      capacity is ``block_threads * decoded_items_per_thread``. Positions
      beyond the sequence contain zero. See
      :ref:`decoding windows and scratch <coop-glossary-decoding>`.

   relative run offset
      A decoded element's position within its own :term:`run`. For runs
      ``[7, 9]`` with lengths ``[3, 2]``, the full sequence has relative
      offsets ``[0, 1, 2, 0, 1]``. This differs from its absolute decoded
      position or its position within a requested window.

   key-value pair
      A key used for ordering or selection and an associated value, such as
      its original array index. Pair operations move the two together.
      Key and value payloads have the same extent but may have different
      dtypes. See :doc:`Merge Sort <visualizations/merge-sort>`.

   stable sort
      A sort that preserves the input order of elements with equal keys.
      A function's contract must promise stability before a program relies
      on it. Radix Sort in ``cuda.coop`` is stable; Merge Sort does not
      promise equal-key order. See :doc:`radix sorting <visualizations/radix>`.

   radix digit
      A fixed-width interval of key bits used in one ranking or sorting
      step. ``radix_rank`` assigns ranks according to one such digit;
      ``radix_sort_keys`` and ``radix_sort_pairs`` order keys over the
      requested bit interval. See :doc:`radix sorting and ranks <visualizations/radix>`.

   common API
      .. raw:: html

         <span id="term-portable-API"></span>

      The backend-independent API exposed through ``from cuda import coop``.
      It describes operations on thread groups, values, and storage.
      Support for particular operations and argument types depends on the
      backend. Qualified APIs provide backend-specific extensions. See
      :ref:`choosing an API <coop-api-namespaces>`.

   qualified API
      A backend's namespace: ``cuda.coop.numba_mlir`` or
      ``cuda.coop.cutlass``. It provides the supported common operations
      and compiler-specific extensions. See
      :ref:`namespace choices <coop-faq-namespaces>`.

   striped
      A layout in which consecutive threads own consecutive tile elements
      at each per-thread item position. For ``G`` threads, thread ``t`` owns
      tile positions ``t``, ``t + G``, ``t + 2 * G``, and so on. See
      :ref:`blocked versus striped <coop-glossary-layouts>`.

   temporary storage
      Shared-memory scratch used internally by a primitive. The compiler
      allocates it automatically when needed. An explicit ``TempStorage``
      descriptor can control allocation sharing and synchronization for
      supported operations. Keep application data in payloads or arrays.
      See :ref:`the temporary-storage FAQ <coop-faq-temp-storage>`.

   thread group
      The threads participating in a primitive, such as a block, a physical
      warp, or a logical group within a warp. A thread's rank identifies its
      position within that group. See :ref:`groups <coop-common-groups>`.

   tile
      The sequence of values processed by one group in an operation. With
      ``G`` threads and ``K`` items per thread, a full tile has ``G * K``
      values. ``valid_items`` in Load and Store selects a prefix of that
      sequence. It counts tile elements, not elements per thread.

   top-k
      Selection of the smallest or largest ``k`` keys, optionally with
      associated values. ``cuda.coop`` TopK returns an unordered selection;
      only its selected prefix is defined. See :doc:`TopK <visualizations/topk>`.

.. _coop-glossary-layouts:

Blocked versus striped
----------------------

Both layouts distribute the same tile across threads. They differ in which
thread owns each position. For four illustrative threads and two items per
thread, a tile of positions ``0`` through ``7`` is distributed as follows:

.. list-table::
   :header-rows: 1

   * - Thread rank
     - Blocked: ``items[0], items[1]``
     - Striped: ``items[0], items[1]``
   * - 0
     - 0, 1
     - 0, 4
   * - 1
     - 2, 3
     - 1, 5
   * - 2
     - 4, 5
     - 2, 6
   * - 3
     - 6, 7
     - 3, 7

For group size ``G``, items per thread ``K``, thread rank ``t``, and local
item slot ``i``:

.. list-table::
   :header-rows: 1

   * - Layout
     - Tile position owned by ``items[i]`` in thread ``t``
     - Owner of tile position ``p``
   * - Blocked
     - ``t * K + i``
     - Thread ``p // K``, slot ``p % K``
   * - Striped
     - ``t + i * G``
     - Thread ``p % G``, slot ``p // G``

The four-thread table illustrates ownership, not a supported physical CUDA
warp size. A physical warp has 32 threads.

Layout describes the values a caller sees. The memory-access algorithm may
use a different intermediate layout. For example, ``load(...,
algorithm="transpose")`` reads striped values, then rearranges them into
blocked payloads. ``algorithm="striped"`` leaves the payloads striped.

Store expects the layout selected by its algorithm. Combining a striped
Load with a direct Store without conversion changes the output order.
``ThreadData`` does not carry a layout tag that corrects this mismatch.

Follow the values in the :doc:`Load <visualizations/load>` and
:doc:`Store <visualizations/store>` visualizations. :doc:`Exchange
<visualizations/exchange>` converts between layouts; the
:ref:`shared layout discussion <coop-common-layouts>` explains how layout
affects Scan.

.. _coop-glossary-decoding:

Run positions, windows, and scratch
-----------------------------------

Run Length Decode first prepares a table of run values and starting
positions in shared scratch. A window offset then selects where to read
from the expanded sequence. For values ``[7, 9]`` and lengths ``[3, 2]``:

.. list-table::
   :header-rows: 1

   * - Absolute decoded position
     - Value
     - Relative run offset
   * - 0
     - 7
     - 0
   * - 1
     - 7
     - 1
   * - 2
     - 7
     - 2
   * - 3
     - 9
     - 0
   * - 4
     - 9
     - 1

A window starting at position two begins with values ``[7, 9, 9]`` and
relative offsets ``[2, 0, 1]``. The total decoded size remains five.
Each independent window call prepares its table again; passing the same
``TempStorage`` descriptor reuses memory, not a prepared decoder.

:func:`cuda.coop.run_length_decode_into` keeps that table alive while it
loops over windows within one call. The table's lifetime ends when the
call returns. Neither form exposes a parent decoder object or a mutable
decode cursor. See :ref:`the RLD lifecycle FAQ <coop-faq-rld-lifecycle>`.

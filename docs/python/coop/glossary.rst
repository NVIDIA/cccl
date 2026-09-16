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
      calls and generate CUDA code for them. Numba-CUDA-MLIR is the first
      supported backend. See :ref:`registration <coop-backend-registration>`.

   blocked
      A layout in which each thread owns consecutive elements of a
      :term:`tile`. With ``K`` items per thread, thread ``t`` owns tile
      positions ``t * K`` through ``t * K + K - 1``. See
      :ref:`blocked versus striped <coop-glossary-layouts>`.

   collective
      An operation executed together by a :term:`thread group`. Every
      required participant must reach the same invocation. See
      :ref:`participation and synchronization <coop-participation>`.

   payload
      The values contributed or received by one thread. ``ThreadData(K)``
      describes a fixed-size payload of ``K`` items. Some operations also
      accept scalars or backend-specific arrays. See
      :ref:`thread data <coop-thread-data>`.

   portable API
      The common API available through ``from cuda import coop``. Its
      operations describe groups, values, and storage independently of a
      particular kernel compiler. Each backend implements its supported
      operations; a common spelling does not guarantee support in every
      compiler. See :ref:`choosing an API <coop-programming-api-choice>`.

   qualified API
      A backend's namespace, such as ``cuda.coop.numba_mlir``. It provides
      the common operations and compiler-specific extensions. See
      :ref:`namespace choices <coop-faq-namespaces>`.

   striped
      A layout in which consecutive threads own consecutive tile elements
      at each per-thread item position. For ``G`` threads, thread ``t`` owns
      tile positions ``t``, ``t + G``, ``t + 2 * G``, and so on. See
      :ref:`blocked versus striped <coop-glossary-layouts>`.

   temporary storage
      Shared-memory scratch used internally by a collective. The compiler
      allocates it automatically when needed. An explicit ``TempStorage``
      descriptor can control allocation sharing and synchronization for
      supported operations. Keep application data in payloads or arrays.
      See :ref:`the temporary-storage FAQ <coop-faq-temp-storage>`.

   thread group
      The threads participating in a collective, such as a block, a physical
      warp, or a logical group within a warp. A thread's rank identifies its
      position within that group. See :ref:`groups <coop-thread-groups>`.

   tile
      The sequence of values processed by one group in an operation. With
      ``G`` threads and ``K`` items per thread, a full tile has ``G * K``
      values. ``valid_items`` in Load and Store selects a prefix of that
      sequence. It counts tile elements, not elements per thread.

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
:ref:`programming guide <coop-data-layouts>` shows how layout affects Scan.

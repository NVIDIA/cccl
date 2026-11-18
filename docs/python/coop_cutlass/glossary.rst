.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-glossary:

Glossary
========

Terms
-----

.. glossary::
   :sorted:

   compiler integration
      The part of the package that works with the CuTe compiler. It records
      each primitive call while CuTe traces a kernel. When tracing ends, it
      compiles the CUB code for all the calls in one NVRTC compilation, and
      CuTe links the result into the kernel. See the
      :doc:`Developer Guide <developer_guide>`.

   blocked
      A layout in which each thread holds consecutive elements of a
      :term:`tile`. With ``K`` items per thread, the thread with
      :term:`rank` ``t`` holds tile positions ``t * K`` to
      ``t * K + K - 1``. See
      :ref:`blocked versus striped <coop-glossary-layouts>`.

   primitive
      .. raw:: html

         <span id="term-collective"></span>

      One of the package's functions that the threads of a
      :term:`thread group` call together, such as ``load``, ``reduce`` or
      ``exclusive_sum``. Every thread of the group makes the call. See
      :ref:`Participation <coop-concepts-participation>` and
      :ref:`the top of the API reference <cuda_coop-module>`.

   single-phase call
      A :term:`primitive` call made in one step, inside the kernel, with
      nothing to construct first. The block size comes from the launch, the
      element type from the memory or the items, and the :term:`scratch`
      from the compiler or from a ``TempStorage`` that the kernel passes.
      CUB's C++ classes take two steps. See
      :ref:`Single-phase calls <coop-concepts-single-phase>`.

   family
      Related :term:`primitives <primitive>` that share a section of the
      :doc:`API reference <api>`. The Scan family is ``scan``,
      ``exclusive_scan``, ``inclusive_scan``, ``exclusive_sum`` and
      ``inclusive_sum``. The :doc:`Developer Guide <developer_guide>` says
      where the code of a family lives.

   payload
      One thread's items, held in a ``ThreadData``. ``load`` returns a
      payload in each thread, and ``ThreadData(K)`` makes a payload of ``K``
      items for the kernel to fill. Primitives other than ``load`` and
      ``store`` also read a CuTe register tensor or a ``TensorSSA`` as a
      payload. A warp scan takes one scalar per thread in place of a
      payload. See :ref:`ThreadData <coop-thread-data>`.

   phi
      An operation in a compiler's intermediate representation (IR) that
      selects a value by the path the program took to reach it. After an
      ``if`` statement assigns different values to a variable, a phi selects
      the value from the branch that ran. At the top of a loop, it selects
      the initial value or the value from the previous trip.

   batch
      One of the reductions that :func:`~cuda.coop.cutlass.reduce_batched`
      makes in one call. Item ``b`` of every lane's payload belongs to batch
      ``b``. A batch covers the lanes of the group, and ``output_layout`` sets
      which lane holds its result. See
      :ref:`Batched reduction <coop-cutlass-reduce-batched>` in the API
      reference and the :doc:`explorer <visualizations/reduce-batched>`.

   bin
      One counter of :func:`~cuda.coop.cutlass.histogram`. A sample with
      value ``b`` adds one to bin ``b``, so every sample must be at least 0
      and less than ``bins``. The counters come back in :term:`striped`
      order: on a block of ``G`` threads, item ``i`` of the thread with rank
      ``t`` holds the count of bin ``t + i * G``. See
      :ref:`Histogram <coop-cutlass-histogram>` in the API reference and the
      :doc:`explorer <visualizations/histogram>`.

   tile boundary
      The edge between one group's tile and the data next to it.
      :func:`~cuda.coop.cutlass.adjacent_difference` and
      :func:`~cuda.coop.cutlass.discontinuity` take the item on the other
      side of the edge as ``tile_predecessor_item`` or
      ``tile_successor_item``. Without it, ``adjacent_difference`` copies
      the item at the edge and ``discontinuity`` flags it with 1. See
      :ref:`Neighbor comparisons <coop-cutlass-neighbors>`.

   run
      A value and the number of times it repeats. The value 7 with the
      length 3 stands for ``7, 7, 7``.
      :func:`~cuda.coop.cutlass.run_length_decode` and
      :func:`~cuda.coop.cutlass.run_length_decode_into` turn a block's runs
      back into the repeated values. In the run lengths of a block, taken
      thread by thread, every positive length comes before every zero, and
      a zero is padding. See
      :ref:`Run Length Decode <coop-cutlass-run-length>` in the API
      reference and the :doc:`explorer <visualizations/run-length-decode>`.

   head flag
      An ``Int32`` flag from :func:`~cuda.coop.cutlass.discontinuity` that
      is 1 where an item differs from the item before it, which marks the
      first item of a run of equal values. Without
      ``tile_predecessor_item``, the tile's first head flag is 1. See
      :ref:`Neighbor comparisons <coop-cutlass-neighbors>`.

   tail flag
      An ``Int32`` flag from :func:`~cuda.coop.cutlass.discontinuity` that
      is 1 where an item differs from the item after it, which marks the
      last item of a run of equal values. Without ``tile_successor_item``,
      the tile's last tail flag is 1. See
      :ref:`Neighbor comparisons <coop-cutlass-neighbors>`.

   decoded window
      The part of the decoded stream that one
      :func:`~cuda.coop.cutlass.run_length_decode` call returns. It starts at
      position ``decoded_window_offset`` of the stream and has
      ``decoded_items_per_thread`` items for each thread of the block. An
      item of the result past the end of the stream is zero.
      :func:`~cuda.coop.cutlass.run_length_decode_into` writes the stream
      one window at a time. See
      :ref:`run positions, windows, and scratch <coop-glossary-decoding>`.

   relative run offset
      A decoded item's position inside its own :term:`run`. For the values
      ``[7, 9]`` with the lengths ``[3, 2]``, the five items of the stream
      have the relative offsets ``[0, 1, 2, 0, 1]`` and the stream
      positions ``[0, 1, 2, 3, 4]``. CUB can return the relative offset of
      every decoded item. ``run_length_decode`` and
      ``run_length_decode_into`` have no such result.

   key-value pair
      A key and a value that travels with it, such as the key's position in
      the input. ``merge_sort_pairs``, ``radix_sort_pairs``,
      ``topk_min_pairs`` and ``topk_max_pairs`` sort or select by the keys
      and move each value with its key. The key payload and the value
      payload have the same number of items per thread, and their element
      types can differ. See
      :ref:`Comparison sorting <coop-cutlass-merge-sort>`,
      :ref:`Radix sorting and ranking <coop-cutlass-radix>` and
      :ref:`Top-k selection <coop-cutlass-topk>`.

   stable sort
      A sort that keeps equal keys in the order they had in the input.
      ``radix_sort_keys`` and ``radix_sort_pairs`` are stable: keys that
      are equal in the bits the sort compares keep their input order.
      ``merge_sort_keys`` and ``merge_sort_pairs`` can return equal keys in
      any order. See :ref:`Radix sorting and ranking <coop-cutlass-radix>`
      and :ref:`Comparison sorting <coop-cutlass-merge-sort>`.

   radix digit
      A few neighboring bits of a key, such as bits 0 to 3.
      ``radix_sort_keys`` and ``radix_sort_pairs`` sort by the bits from
      ``begin_bit`` to ``end_bit - 1``, one digit at a time from the lowest
      up. ``radix_rank_keys`` ranks the keys by one digit of ``radix_bits``
      bits. See :ref:`Radix sorting and ranking <coop-cutlass-radix>` in the
      API reference and the :doc:`explorer <visualizations/radix>`.

   striped
      A layout in which neighboring threads hold neighboring elements of a
      :term:`tile`. In a group of ``G`` threads, the thread with
      :term:`rank` ``t`` holds tile positions ``t``, ``t + G``,
      ``t + 2 * G`` and so on. See
      :ref:`blocked versus striped <coop-glossary-layouts>`.

   scratch
      Shared memory that a :term:`primitive` uses while it runs. A block
      scan needs it, and a ``load`` with the default algorithm needs none.
      Leave ``temp_storage`` out of a call and the compiler provides the
      scratch and a barrier after the call. Pass a ``TempStorage`` to choose
      which calls share bytes and where the barriers go. See
      :ref:`TempStorage <coop-temp-storage>`.

   temporary storage
      CUB's name for :term:`scratch`. A
      :class:`~cuda.coop.cutlass.TempStorage` is one region of shared memory
      that holds the scratch of every call it is passed to.
      :meth:`~cuda.coop.cutlass.TempStorage.reserve` returns a typed shared
      tensor inside the region for the kernel's own data. The kernel places
      the barriers for that tensor, and with the default ``sharing`` the
      next primitive that uses the region overwrites it. See
      :ref:`TempStorage <coop-temp-storage>`.

   thread group
      The threads that make a call together. The first argument of every
      :term:`primitive` names them: ``this_block()``, ``this_warp()`` or
      ``this_warp().group_by(n)``. See
      :ref:`Thread groups <coop-thread-groups>`.

   lane
      One thread of a warp. A warp has 32 lanes, and ``this_warp().rank()``
      returns the calling thread's lane, from 0 to 31. See
      :ref:`Thread groups <coop-thread-groups>`.

   logical warp
      CUB's name for a group of lanes. ``this_warp().group_by(n)`` is the
      calling thread's group of ``n`` neighboring :term:`lanes <lane>`, so
      ``group_by(8)`` splits a warp into four groups of eight. See
      :ref:`Thread groups <coop-thread-groups>`.

   rank
      A thread's position in its :term:`thread group`, counted from 0.
      ``group.rank()`` returns it. ``reduce`` and ``sum`` leave their result
      in the thread with rank 0. ``radix_rank_keys`` and the ``ranks`` of
      ``exchange`` use the word for a position among the items of a
      :term:`tile`. See :ref:`Thread groups <coop-thread-groups>`.

   tile
      The elements one group works on in one call. A group of ``G`` threads
      with ``K`` items per thread has a tile of ``G * K`` elements. For
      ``load`` and ``store`` they are consecutive elements of memory, and
      ``valid_items`` counts them from the start of the tile, from 0 to
      ``G * K``. See :ref:`Load & Store <memory-operations>`.

   top-k
      Selection of up to ``k`` smallest or largest keys, optionally with a
      value for each key. The selected keys fill the first positions of the
      result in no defined order. See
      :ref:`Top-k selection <coop-cutlass-topk>` in the API reference and
      the :doc:`explorer <visualizations/topk>`.

.. _coop-glossary-layouts:

Blocked versus striped
----------------------

Both layouts spread one :term:`tile` over the threads of a group. They
differ in which thread holds each element. With four threads and two items
per thread, the tile positions 0 to 7 go to these threads:

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

For a group of ``G`` threads with ``K`` items each, where ``t`` is a
thread's rank and ``i`` is the index of one of its items:

.. list-table::
   :header-rows: 1

   * - Layout
     - Tile position of ``items[i]`` in thread ``t``
     - Holder of tile position ``p``
   * - Blocked
     - ``t * K + i``
     - Thread ``p // K``, item ``p % K``
   * - Striped
     - ``t + i * G``
     - Thread ``p % G``, item ``p // G``

The first table shows four threads to stay small.
``this_warp().group_by(4)``, a group of four :term:`lanes <lane>`, has
these layouts.

A layout describes the payload that a call returns or takes. Inside the
call the items can pass through another order. ``load`` with
``algorithm="transpose"`` reads the tile in striped order and then exchanges
the items between the threads, so it returns a blocked payload.
``algorithm="striped"`` returns the striped payload.

A payload keeps no record of its layout, and ``store`` reads it in the
layout of its own ``algorithm``. In a group of more than one thread with
more than one item each, a tile loaded with ``"striped"`` and stored with
``"direct"`` is written in another order.

The :doc:`Load <visualizations/load>` and :doc:`Store
<visualizations/store>` explorers follow every element.
:func:`~cuda.coop.cutlass.exchange` converts one layout into the other, and
the :doc:`Exchange <visualizations/exchange>` explorer shows it. A scan
reads a payload in blocked order, as
:ref:`Layouts <coop-concepts-layouts>` explains.

.. _coop-glossary-decoding:

Run positions, windows, and scratch
-----------------------------------

A :term:`run` starts at the position of the decoded stream where the runs
before it end. For the values ``[7, 9]`` with the lengths ``[3, 2]``, the
first run starts at position 0 and the second at position 3:

.. list-table::
   :header-rows: 1

   * - Position in the decoded stream
     - Value
   * - 0
     - 7
   * - 1
     - 7
   * - 2
     - 7
   * - 3
     - 9
   * - 4
     - 9

A :term:`decoded window` that starts at position 2 begins with ``7, 9, 9``.
The stream still has five items.

Run Length Decode keeps the value and the starting position of each run in
a table in :term:`scratch`, and reads a window from that table. Each
``run_length_decode`` call makes its own table, even when the calls share a
``TempStorage``: they reuse its bytes and nothing else. One
:func:`~cuda.coop.cutlass.run_length_decode_into` call makes one table and
reads all its windows from it. The table lasts until the call returns. A
call with nothing to decode makes no table. Neither function returns a
decoder or keeps a position for a later call to use. See
:ref:`Run Length Decode <coop-cutlass-run-length>` in the API reference.

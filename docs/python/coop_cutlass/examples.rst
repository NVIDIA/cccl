.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Executable examples
===================

Each program in ``python/cuda_coop_cutlass/examples/`` is complete and checks
its own output. Install the package and its
:ref:`requirements <coop-cutlass-requirements>`, then run a program from the
repository root:

.. code-block:: bash

   python python/cuda_coop_cutlass/examples/prefix_sum.py

The command calls the file's ``run_example`` function with its default
arguments. ``run_example`` allocates memory on the device, compiles and
launches the kernel, and checks the output against NumPy on the host. A
failed check raises an error. A run that ends without an error passed every
check. Only ``tensor_memory_topk.py`` prints a line of its own when it
passes.

The table lists every program, ``prefix_sum.py`` first and the rest in the
order of the :doc:`API reference <api>`:

.. list-table::
   :header-rows: 1
   :widths: 26 42 32

   * - File
     - Shows
     - Primitives
   * - `prefix_sum.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/prefix_sum.py>`__
     - A running sum over one block
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.inclusive_sum`,
       :func:`~cuda.coop.cutlass.store`
   * - `block_storage.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/block_storage.py>`__
     - One :class:`~cuda.coop.cutlass.TempStorage` reused on every trip
       around a loop
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.store`
   * - `shared_workspace.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_workspace.py>`__
     - The bytes of a tensor from
       :meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>`,
       used in turn by native CuTe code and as scratch by ``sum``
     - :func:`~cuda.coop.cutlass.sum`
   * - `block_load_store.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/block_load_store.py>`__
     - A partial :term:`tile` on a block: ``valid_items`` and ``offset`` in both
       calls, and in ``load`` an ``oob_default`` for the items at or past
       the count
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.store`
   * - `warp_load_store.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/warp_load_store.py>`__
     - One tile for each of a block's two warps, each with its own
       ``valid_items``
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.store`
   * - `logical_warp_load_store.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/logical_warp_load_store.py>`__
     - One tile for each of eight groups of eight lanes, each with its own
       ``valid_items``
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.store`
   * - `shared_staging.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_staging.py>`__
     - A tile staged in shared memory by a native CuTe copy, then loaded
       from it
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.store`
   * - `reduce.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/reduce.py>`__
     - Reductions over a block, over groups of eight lanes and over the
       first 23 threads
     - :func:`~cuda.coop.cutlass.sum`,
       :func:`~cuda.coop.cutlass.reduce`
   * - `reduce_batched.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/reduce_batched.py>`__
     - One sum for each item of a lane's :term:`payload`
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.reduce_batched`
   * - `scan.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/scan.py>`__
     - An exclusive scan with an initial value, and scratch shared with
       Load and Store
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.exclusive_scan`,
       :func:`~cuda.coop.cutlass.store`
   * - `exchange_shuffle.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/exchange_shuffle.py>`__
     - A :term:`blocked` payload made :term:`striped`, then shifted by one
       position
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.exchange`,
       :func:`~cuda.coop.cutlass.shuffle`,
       :func:`~cuda.coop.cutlass.store`
   * - `merge_sort.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/merge_sort.py>`__
     - Keys and key-value pairs sorted over the first part of a tile
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.merge_sort_keys`,
       :func:`~cuda.coop.cutlass.merge_sort_pairs`,
       :func:`~cuda.coop.cutlass.store`
   * - `radix.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/radix.py>`__
     - A sort, a pair sort on four bits, and ranks on the same bits
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.radix_sort_keys`,
       :func:`~cuda.coop.cutlass.radix_sort_pairs`,
       :func:`~cuda.coop.cutlass.radix_rank_keys`,
       :func:`~cuda.coop.cutlass.store`
   * - `topk.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/topk.py>`__
     - The smallest keys and the largest key-value pairs of the first part
       of a tile
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.topk_min_keys`,
       :func:`~cuda.coop.cutlass.topk_max_pairs`,
       :func:`~cuda.coop.cutlass.store`
   * - `tensor_memory_topk.py
       <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/tensor_memory_topk.py>`__
     - Top-k selection between Tensor Core matrix products, on an SM100 or
       SM103 GPU
     - :func:`~cuda.coop.cutlass.topk_max_keys`

Ten of the programs are listed in full below, in the same order.

prefix_sum.py
-------------

This program computes the running sum of 512 integers with one block of 128
threads. :ref:`A first program <coop-cutlass-load-store>` in the Programming
Guide explains its parts.

`View prefix_sum.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/prefix_sum.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/prefix_sum.py
   :language: python

shared_staging.py
-----------------

In this program each group of eight lanes stages its tile: a native CuTe
copy brings it from global memory into shared memory. After a block barrier,
``load`` reads it from there.
:ref:`Staging a tile in shared memory <coop-shared-load-store>` in Advanced
topics explains the copy and the barrier.

`View shared_staging.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_staging.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_staging.py
   :language: python

reduce.py
---------

This program computes three reductions on a block of 64 threads: the sum of
every item, the largest ``payload[0]`` in each group of eight lanes, and the
sum of ``payload[0]`` over the first 23 threads.
:ref:`Reduction <coop-cutlass-reduce>` in the API reference explains each
call.

`View reduce.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/reduce.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce.py
   :language: python

reduce_batched.py
-----------------

This program computes three sums in each of two warps, one for each item of
a lane's payload. :ref:`Batched reduction <coop-cutlass-reduce-batched>` in
the API reference explains which lane holds each sum.

`View reduce_batched.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/reduce_batched.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce_batched.py
   :language: python

scan.py
-------

This program runs :func:`~cuda.coop.cutlass.exclusive_scan` with an initial
value of 7 over the 128 items of one block, and its Load, Scan and Store
share one :ref:`TempStorage <coop-temp-storage>`.

`View scan.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/scan.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/scan.py
   :language: python

exchange_shuffle.py
-------------------

This program converts a block's payload from blocked to striped with
:func:`~cuda.coop.cutlass.exchange`, shifts it down by one position with
:func:`~cuda.coop.cutlass.shuffle`, and sets to zero the one item that the
shift leaves undefined.

`View exchange_shuffle.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/exchange_shuffle.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py
   :language: python

merge_sort.py
-------------

This program sorts the first 185 of a block's 192 keys in ascending order
with :func:`~cuda.coop.cutlass.merge_sort_keys`, and again in descending
order with :func:`~cuda.coop.cutlass.merge_sort_pairs`, which carries a
value with each key. It stores both inputs as well, to show that they keep
their values.

`View merge_sort.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/merge_sort.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/merge_sort.py
   :language: python

radix.py
--------

This program sorts the 128 signed keys of one block with
:func:`~cuda.coop.cutlass.radix_sort_keys`. It sorts them again with
:func:`~cuda.coop.cutlass.radix_sort_pairs`, which carries each key's
position in the tile as its value. That sort is descending, compares only
the four highest bits, and returns a striped result.
:func:`~cuda.coop.cutlass.radix_rank_keys` then ranks the keys by the same
four bits, also in descending order. Its ``exclusive_digit_prefix`` argument
receives, for each of the 16 values of those bits, the number of keys ranked
before the keys with that value. The keys are signed, so every call compares
them with the sign bit flipped.

`View radix.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/radix.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/radix.py
   :language: python

topk.py
-------

This program selects the 31 smallest keys with
:func:`~cuda.coop.cutlass.topk_min_keys` and the 31 largest key-value pairs
with :func:`~cuda.coop.cutlass.topk_max_pairs`, from the first 93 items of a
block's tile. Its check on the host accepts the selected keys in any order
and any choice among equal keys.

`View topk.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/topk.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/topk.py
   :language: python

.. _tensor-memory-topk-py:

tensor_memory_topk.py
---------------------

This program alternates Tensor Core matrix products with
:func:`~cuda.coop.cutlass.topk_max_keys`: after each product the threads
load the accumulator from tensor memory into registers, select its largest
values, and write the selection back for the next product. It needs an SM100
or SM103 GPU. The :doc:`Tensor Core and TopK walkthrough <tensor_memory_topk>`
explains the copy layouts, the synchronization and how each selection feeds
the next product.

`View tensor_memory_topk.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/tensor_memory_topk.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/tensor_memory_topk.py
   :language: python

Integration walkthroughs
------------------------

The :ref:`debugger walkthrough <cuda.coop.cutlass.debugger_walkthrough>`
follows a tile copy through a Python debugger, once with
``algorithm="direct"`` and once with ``algorithm="transpose"``. In each run
its script compiles the copy kernel once and launches it twice. The
walkthrough shows excerpts and links the
`complete debugger_walkthrough.py script on GitHub
<https://github.com/NVIDIA/cccl/blob/main/docs/python/coop_cutlass/debugger_walkthrough.py>`_.

The :doc:`minimal CuTe DSL integration example <cutlass_integration_example>`
calls ``cub::BlockReduce`` from a CuTe kernel using the DSL's own interfaces.
Its walkthrough explains each step with code excerpts and offers the
`complete cutlass_integration_example.py script on GitHub
<https://github.com/NVIDIA/cccl/blob/main/docs/python/coop_cutlass/cutlass_integration_example.py>`_.

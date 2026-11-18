.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda_coop-module:
.. _coop-cutlass-extensions:

``cuda.coop.cutlass`` API Reference
===================================

.. warning::
   This is an experimental API and is subject to change.

.. py:module:: cuda.coop.cutlass

.. automodule:: cuda.coop.cutlass
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. _coop-thread-groups:
.. _coop-cutlass-hierarchy:

Thread groups
-------------

.. automodule:: cuda.coop.cutlass._thread_group
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: this_block
.. autofunction:: this_warp
.. autofunction:: this_thread
.. autofunction:: this_cluster
.. autofunction:: this_grid

.. autoclass:: ThreadGroup
   :no-members:
   :no-special-members:

   .. automethod:: group_by
   .. automethod:: rank
   .. automethod:: count
   .. automethod:: rank_as
   .. automethod:: count_as
   .. automethod:: is_member
   .. automethod:: sync
   .. automethod:: sync_aligned

.. autoclass:: ThreadHierarchy
   :no-members:
   :no-special-members:

.. py:class:: Hierarchy

   Alias for :class:`ThreadHierarchy`.

.. _coop-thread-data:
.. _coop-cutlass-register-payloads:
.. _coop-cutlass-dtype-inference:

ThreadData: per-thread payloads and CuTe conversion
---------------------------------------------------

.. automodule:: cuda.coop.cutlass._thread_data
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autoclass:: ThreadData
   :no-members:
   :no-special-members:

.. automethod:: ThreadData.from_values
.. automethod:: ThreadData.from_fn
.. automethod:: ThreadData.from_register_tensor
.. automethod:: ThreadData.from_vector
.. automethod:: ThreadData.from_payload
.. automethod:: ThreadData.load
.. automethod:: ThreadData.to_tensor_ssa
.. automethod:: ThreadData.to_register_tensor

.. _coop-temp-storage:

TempStorage: temporary storage
------------------------------

.. automodule:: cuda.coop.cutlass._temp_storage
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autoclass:: TempStorage
   :no-members:
   :no-special-members:

.. automethod:: TempStorage.sync
.. automethod:: TempStorage.reserve
.. automethod:: TempStorage.release

.. _memory-operations:

Load & Store
------------

.. automodule:: cuda.coop.cutlass._group_load_store
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: load
.. autofunction:: store

.. _coop-cutlass-reduce:

Reduction
---------

.. automodule:: cuda.coop.cutlass._group_reduce
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: reduce
.. autofunction:: sum

.. _coop-cutlass-reduce-batched:

.. automodule:: cuda.coop.cutlass._group_reduce_batched
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: reduce_batched

.. _coop-cutlass-scan:

Scan
----

.. automodule:: cuda.coop.cutlass._group_scan
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: scan
.. autofunction:: exclusive_sum
.. autofunction:: inclusive_sum
.. autofunction:: exclusive_scan
.. autofunction:: inclusive_scan

.. _coop-cutlass-exchange:

Data rearrangement
------------------

.. automodule:: cuda.coop.cutlass._group_exchange
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: exchange

.. _coop-cutlass-shuffle:

.. automodule:: cuda.coop.cutlass._group_shuffle
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: shuffle

.. _coop-cutlass-merge-sort:

Comparison sorting
------------------

.. automodule:: cuda.coop.cutlass._group_merge_sort
   :no-members:
   :no-special-members:
   :no-index:

.. currentmodule:: cuda.coop.cutlass

.. autofunction:: merge_sort_keys
.. autofunction:: merge_sort_pairs

Radix sorting and ranking
-------------------------

.. autofunction:: radix_sort_keys
.. autofunction:: radix_sort_pairs
.. autofunction:: radix_rank_keys

Top-k selection
---------------

.. autofunction:: topk_min_keys
.. autofunction:: topk_max_keys
.. autofunction:: topk_min_pairs
.. autofunction:: topk_max_pairs

Neighbor comparisons
--------------------

.. autofunction:: adjacent_difference
.. autofunction:: discontinuity

Histogram
---------

.. autofunction:: histogram

Run Length Decode
-----------------

.. autofunction:: run_length_decode
.. autofunction:: run_length_decode_into

Compiled device operators
-------------------------

.. autofunction:: compile_operator

.. autoclass:: DeviceOperator
   :no-members:
   :no-special-members:

Producer protocols
------------------

.. autoclass:: ThreadDataLoadSource
   :no-members:
   :no-special-members:

.. autoclass:: ThreadDataTensorMetadata
   :no-members:
   :no-special-members:

.. autoclass:: ThreadDataSource
   :no-members:
   :no-special-members:

See :ref:`producer interoperability <coop-cutlass-producer-interop>` for
capability detection, copy ownership, and lifetime requirements.

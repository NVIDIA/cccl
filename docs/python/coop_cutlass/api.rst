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
.. currentmodule:: cuda.coop.cutlass

Use ``import cuda.coop.cutlass as coop`` inside programs compiled with CuTe.
The installed ``.pyi`` files describe overload and result typing. The
:doc:`Programming Guide <programming_guide>` explains payload ownership,
participation, layouts, and operation-specific controls.

Load always returns a fresh ``ThreadData``, including when
``items_per_thread=1``. Global and shared tensors can be Load sources or
Store destinations. CuTe register and immutable-vector adapters provide the
conversion rules for operations accepting those forms.

Reduce and Sum return CuTe scalars, even when a NumPy dtype selects the
payload type; stubs report that NumPy type for static checks. Array operations
return ``ThreadData`` unless their documented contract says otherwise. Warp
Scan accepts one scalar per lane. Rank results and Discontinuity flags have
signed ``Int32`` dtype. Bulk Run Length Decode returns a ``Uint32`` total to
every block member.

Recognized ``operator`` functions and NumPy ufuncs select built-in operators.
Other callables, custom Reduce and Scan operators, and Scan prefix callbacks
are unsupported.

Thread groups
-------------

.. autofunction:: this_thread
.. autofunction:: this_warp
.. autofunction:: this_block
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

Per-thread payloads and CuTe conversion
---------------------------------------

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

Temporary storage
-----------------

.. autoclass:: TempStorage
   :no-members:
   :no-special-members:

.. automethod:: TempStorage.reserve
.. automethod:: TempStorage.release

Memory operations
^^^^^^^^^^^^^^^^^

.. autofunction:: load
.. autofunction:: store

Reduction
^^^^^^^^^

.. autofunction:: reduce
.. autofunction:: reduce_batched
.. autofunction:: sum

Scan
^^^^

.. autofunction:: scan
.. autofunction:: exclusive_sum
.. autofunction:: inclusive_sum
.. autofunction:: exclusive_scan
.. autofunction:: inclusive_scan

Data rearrangement
^^^^^^^^^^^^^^^^^^

.. autofunction:: exchange
.. autofunction:: shuffle

Comparison sorting
^^^^^^^^^^^^^^^^^^

.. autofunction:: merge_sort_keys
.. autofunction:: merge_sort_pairs

Radix sorting and ranking
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: radix_sort_keys
.. autofunction:: radix_sort_pairs
.. autofunction:: radix_rank_keys

Top-k selection
^^^^^^^^^^^^^^^

.. autofunction:: topk_min_keys
.. autofunction:: topk_max_keys
.. autofunction:: topk_min_pairs
.. autofunction:: topk_max_pairs

Neighbor comparisons
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: adjacent_difference
.. autofunction:: discontinuity

Histogram
^^^^^^^^^

.. autofunction:: histogram

Run Length Decode
^^^^^^^^^^^^^^^^^

.. autofunction:: run_length_decode
.. autofunction:: run_length_decode_into

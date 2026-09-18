.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

.. warning::
   ``cuda.coop`` is an experimental API and is subject to change.

.. _coop-portable-api:

Portable API
------------

The collective functions below are compiler markers; ``register`` is a
host-side configuration function. The installed ``.pyi``
files are authoritative for overload and result typing.

.. currentmodule:: cuda.coop

Backend registration
^^^^^^^^^^^^^^^^^^^^

See :ref:`registering a backend <coop-backend-registration>`.

.. autofunction:: register

Thread groups
^^^^^^^^^^^^^

See :ref:`thread groups <coop-thread-groups>` and
:ref:`participation and synchronization <coop-participation>` for the shared
execution model. A descriptor's availability does not imply that every
collective supports that group.

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

Payloads and temporary storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ThreadData

.. autoclass:: ThreadDataLike
   :no-members:
   :no-special-members:

.. autofunction:: TempStorage

.. autoclass:: TempStorageLike
   :no-members:
   :no-special-members:

Memory operations
^^^^^^^^^^^^^^^^^

.. autofunction:: load
.. autofunction:: store

Reduction
^^^^^^^^^

See :ref:`reduction and result ownership <coop-reductions>`.

.. autofunction:: reduce
.. autofunction:: sum

Scan
^^^^

See :ref:`scan operators and prefixes <coop-scans>`.

.. autofunction:: scan
.. autofunction:: exclusive_sum
.. autofunction:: inclusive_sum
.. autofunction:: exclusive_scan
.. autofunction:: inclusive_scan

Data rearrangement
^^^^^^^^^^^^^^^^^^

See :ref:`blocked and striped layouts <coop-data-layouts>`.

.. autofunction:: exchange
.. autofunction:: shuffle

Comparison sorting
^^^^^^^^^^^^^^^^^^

See :ref:`sorting keys and associated values <coop-merge-sort>`.

.. autofunction:: merge_sort_keys
.. autofunction:: merge_sort_pairs

Radix sorting and ranking
^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`radix sorting and digit ranks <coop-radix>`.

.. autofunction:: radix_sort_keys
.. autofunction:: radix_sort_pairs
.. autofunction:: radix_rank

Top-k selection
^^^^^^^^^^^^^^^

See :ref:`selecting the smallest or largest keys <coop-topk>`.

.. autofunction:: topk_min_keys
.. autofunction:: topk_max_keys
.. autofunction:: topk_min_pairs
.. autofunction:: topk_max_pairs

.. _coop-numba-extensions:

Numba-CUDA-MLIR-qualified API
-----------------------------

.. py:module:: cuda.coop.numba_mlir

Use this module for the extensions below. Shared parameters and behavior
follow the :ref:`Portable API <coop-portable-api>`.

.. code-block:: python

   from cuda.coop import numba_mlir as numba_coop

Qualified calls also accept fixed-size, one-dimensional local arrays where
the operation accepts per-thread payloads. ``local`` and ``shared`` expose
Numba-CUDA-MLIR's memory namespaces.

.. currentmodule:: cuda.coop.numba_mlir

Reduction
^^^^^^^^^

.. autofunction:: reduce

Scan
^^^^

.. autofunction:: scan
.. autofunction:: exclusive_sum
.. autofunction:: inclusive_sum
.. autofunction:: exclusive_scan
.. autofunction:: inclusive_scan

.. autoclass:: StatefulFunction
   :no-members:
   :no-special-members:

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
.. autofunction:: radix_rank

Top-k selection
^^^^^^^^^^^^^^^

.. autofunction:: topk_min_keys
.. autofunction:: topk_max_keys
.. autofunction:: topk_min_pairs
.. autofunction:: topk_max_pairs

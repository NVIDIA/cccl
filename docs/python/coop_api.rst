.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

.. warning::
   ``cuda.coop`` is an experimental API and is subject to change.

.. _coop-portable-api:
.. _portable-api:
.. _coop-common-api:

Common API
----------

The primitive functions below are compiler markers; ``register`` is a
host-side configuration function. The installed ``.pyi``
files are authoritative for overload and result typing. See
:ref:`backend coverage <coop-backends>` for implemented families; a common
entry point does not imply support in every compiler.

.. currentmodule:: cuda.coop

Backend registration
^^^^^^^^^^^^^^^^^^^^

See :ref:`registering a backend <coop-backend-registration>`.

.. autofunction:: register

Thread groups
^^^^^^^^^^^^^

See :ref:`thread groups <coop-common-groups>` and
:ref:`participation and synchronization <coop-common-participation>` for the shared
execution model. A descriptor's availability does not imply that every
primitive supports that group.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

See :ref:`reduction and result ownership <coop-common-results>`.

.. autofunction:: reduce
.. autofunction:: sum
.. autofunction:: reduce_batched

Scan
^^^^

See Scan in the :ref:`Numba guide <coop-scans>` and
:ref:`CUTLASS guide <coop-cutlass-scan>`.

.. autofunction:: scan
.. autofunction:: exclusive_sum
.. autofunction:: inclusive_sum
.. autofunction:: exclusive_scan
.. autofunction:: inclusive_scan

Data rearrangement
^^^^^^^^^^^^^^^^^^

See :ref:`blocked and striped layouts <coop-common-layouts>`.

.. autofunction:: exchange
.. autofunction:: shuffle

Comparison sorting
^^^^^^^^^^^^^^^^^^

See the :ref:`Numba <coop-merge-sort>` and
:ref:`CUTLASS <coop-cutlass-merge-sort>` Merge Sort examples.

.. autofunction:: merge_sort_keys
.. autofunction:: merge_sort_pairs

Radix sorting and ranking
^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`the Numba Radix Sort and Rank examples <coop-radix>`.

.. autofunction:: radix_sort_keys
.. autofunction:: radix_sort_pairs
.. autofunction:: radix_rank

Top-k selection
^^^^^^^^^^^^^^^

See :ref:`the Numba TopK examples <coop-topk>`.

.. autofunction:: topk_min_keys
.. autofunction:: topk_max_keys
.. autofunction:: topk_min_pairs
.. autofunction:: topk_max_pairs

Neighbor comparisons
^^^^^^^^^^^^^^^^^^^^

See :doc:`neighbor operations <coop/neighbor-operations>` for tile boundaries
and the difference between arithmetic results and flags.

.. autofunction:: adjacent_difference
.. autofunction:: discontinuity

Histogram
^^^^^^^^^

See the :doc:`Histogram visualization <coop/visualizations/histogram>`.

.. autofunction:: histogram

Run Length Decode
^^^^^^^^^^^^^^^^^

See :doc:`windowed and bulk decoding <coop/visualizations/run-length-decode>`.

.. autofunction:: run_length_decode
.. autofunction:: run_length_decode_into

.. _coop-numba-extensions:

Numba-CUDA-MLIR-qualified API
-----------------------------

.. py:module:: cuda.coop.numba_mlir

Use this module for the extensions below. Shared parameters and behavior
follow the :ref:`Common API <coop-common-api>`. The
:ref:`comparison table <coop-programming-api-choice>` in the
:doc:`Numba-CUDA-MLIR Programming Guide <coop/programming_guide>` explains
when to choose qualified calls. The
:doc:`Numba-CUDA-MLIR Developer Guide <coop/developer_overview>` follows
their compiler implementation.

.. code-block:: python

   import cuda.coop.numba_mlir as coop

Qualified calls also accept fixed-size, one-dimensional local arrays where
the operation accepts per-thread payloads. ``local`` and ``shared`` expose
Numba-CUDA-MLIR's memory namespaces.

.. currentmodule:: cuda.coop.numba_mlir

Reduction
^^^^^^^^^

.. autofunction:: reduce
.. autofunction:: reduce_batched

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

.. _coop-cutlass-extensions:

CUTLASS-qualified API
---------------------

.. py:module:: cuda.coop.cutlass

Shared parameters and behavior follow the
:ref:`Common API <coop-common-api>`. The
:ref:`comparison table <coop-cutlass-api-choice>` in the
:doc:`CUTLASS Programming Guide <coop_cutlass>` describes the extensions
below and provides executable examples. See the
:doc:`CUTLASS Developer Guide <coop/cutlass_developer_guide>` for compiler
ownership, providers, linking, and storage allocation.

.. list-table:: CUTLASS extensions
   :header-rows: 1
   :widths: 24 76

   * - API
     - Qualified behavior
   * - ``ThreadData``
     - Conversions to and from CuTe register tensors and immutable register
       values; see :ref:`register payloads <coop-cutlass-register-payloads>`.
   * - Reduce and Scan operators
     - Recognized ``operator`` and NumPy aliases for built-in operators.
       Arbitrary device callbacks are unsupported.
   * - Scan and Sum
     - Warp valid-prefix and writable aggregate-output controls; see
       :ref:`Scan <coop-cutlass-scan>`.
   * - Exchange
     - Block warp-striped layouts, scatter ranks and flags, and
       ``warp_time_slicing``; see :ref:`Exchange <coop-cutlass-exchange>`.
   * - Shuffle
     - Scalar Offset and Rotate modes with integer distances; see
       :ref:`Shuffle <coop-cutlass-shuffle>`.
   * - Merge Sort
     - CuTe register-tensor inputs return fresh ``ThreadData`` payloads;
       controls otherwise follow the common API. See
       :ref:`Merge Sort <coop-cutlass-merge-sort>`.

Custom operators and Scan prefix callbacks are not supported. See
:ref:`CUTLASS-specific behavior and limits <coop-cutlass-differences>` and
:ref:`backend coverage <coop-backends>` before selecting a family. The
installed ``.pyi`` files declare supported signatures.

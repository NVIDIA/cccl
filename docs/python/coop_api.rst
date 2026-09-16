.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

.. warning::
   ``cuda.coop`` is an experimental API and is subject to change.

Portable API
------------

The portable functions below are compiler markers. The installed ``.pyi``
files are authoritative for overload and result typing.

.. currentmodule:: cuda.coop

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

.. _coop-numba-extensions:

Numba-CUDA-MLIR-qualified API
-----------------------------

.. py:module:: cuda.coop.numba_mlir

The qualified module provides matching Block, physical-Warp, and logical-Warp
Load, Store, Exchange, and Scan entry points; block Shuffle; hierarchy-aware
Reduce; group descriptors; ``ThreadData``; ``TempStorage``; and
``StatefulFunction``. It additionally exposes backend memory namespaces. Both
constructors accept the portable ``alignment`` keyword for minimum payload
storage alignment. Portable and qualified calls use the same lowercase string
selectors. Block Load and Store support ``direct``, ``striped``,
``vectorize``, ``transpose``, ``warp_transpose``, and
``warp_transpose_timesliced``. Physical and logical Warp calls support
``direct``, ``striped``, ``vectorize``, and ``transpose``. Use ``this_warp()``
for the physical width of 32 or ``this_warp().group_by(width)`` for a logical
width of 1, 2, 4, 8, 16, or 32. The enclosing block must contain a multiple of
32 threads.

``direct``, ``striped``, and ``vectorize`` are storage-free at both scopes.
Warp ``transpose`` uses compiler-owned storage with one disjoint slice per
physical or logical group and a masked ``syncwarp`` reuse barrier. Explicit
``TempStorage`` is rejected for every Warp algorithm. Transpose Store
operations preserve their caller-owned input payload while CUB performs its
internal reordering.

Exchange returns a fresh payload and preserves its input. The portable modes
are ``striped_to_blocked`` and ``blocked_to_striped``. The qualified API adds
block-only warp-striped and scatter layouts, signed rank payloads, non-boolean
integer validity flags, and warp time slicing. Physical and logical Warp
Exchange retain the two portable modes. Shuffle returns a fresh block payload
for unit ``up`` and ``down`` modes, or a scalar for the qualified ``offset`` and
``rotate`` modes. Boundary-output projections are not exposed.

Scan exposes ``scan``, ``exclusive_scan``, ``inclusive_scan``,
``exclusive_sum``, and ``inclusive_sum``. Block Scan accepts scalars and fixed
per-thread arrays and supports ``raking``, ``raking_memoize``, and
``warp_scans``. Warp Scan accepts one scalar per lane. The qualified API adds
stateless operator callbacks, a one-item ``aggregate_output``, Warp-only
``valid_items``, and Block-only prefix callbacks.

Every qualified Block Scan spelling also accepts a block-prefix callback with
the ``prefix_op`` keyword. A stateless callback receives the block aggregate
and returns the prefix. A stateful callback is wrapped in ``StatefulFunction``
and receives a one-item state payload followed by the block aggregate; that
state is passed as the third positional argument. It must be a numeric
one-item ``ThreadData`` or local array whose dtype exactly matches the
descriptor dtype, although that dtype may differ from the scanned value dtype.

Prefix callbacks are qualified-only and Block-only. They cannot be combined
with ``initial_value`` or ``aggregate_output``. They are not stateful binary
scan operators, do not add Warp or ``valid_items`` support, and do not accept
structured state. CUB may invoke the callback in every lane of the block's
first warp, but only lane 0's returned prefix is applied. Initialize every
thread's state cell identically before the first collective and treat thread
0's state as authoritative after repeated calls. Prefix callbacks retain the
normal Block Scan ``TempStorage`` contract. Repeated calls that reuse storage
must retain the automatic block barrier or execute ``syncthreads`` after each
call when a caller-owned descriptor sets ``auto_sync=False``.

For Warp Load and Store, each group receives an automatic memory origin of
``group_index * (group_size * items_per_thread)`` before the caller's element
offset is applied, where the index is the x-major linear thread rank divided by
the group size. Its ``valid_items`` count is relative to that group tile.

``ThreadGroup`` exposes ``rank``, ``count``, ``rank_as``, ``count_as``,
``is_member``, ``sync``, and ``sync_aligned`` with the C++ hierarchy semantics.
Queries accept the thread, Warp, block, cluster, and grid levels. Default query
results use the C++ unsigned product type, normally ``uint32`` and ``uint64``
when the group or queried outer level is the grid; ``*_as`` accepts explicit
signed or unsigned 8-, 16-, 32-, and 64-bit integer dtypes.

``group_by`` accepts only compile-time ``count`` and ``exhaustive`` values.
Mapped groups may query their constituents and immediate physical parent but
not a higher level. Mapped groups of physical warps support queries, membership, and the
restricted reductions described by :func:`cuda.coop.reduce`; their explicit
synchronization methods are rejected. Grid
synchronization is also rejected because this backend does not request a
cooperative grid launch. Callers of non-exhaustive partitions should use
``is_member()`` to guard rank-dependent work for excluded threads, but must not
skip a collective unless that collective's participation contract permits it.
All supported synchronization calls require the participating group to
converge; ``sync_aligned`` additionally requires an aligned group.

See the :github:`Numba-CUDA-MLIR type declarations
<python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi>` for the complete overload
contract. Importing this qualified module requires the matching
Numba-CUDA-MLIR extra.

Qualified function reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The qualified functions retain the common parameter order and add the
extensions described below. The group factories and descriptors follow the
same :ref:`thread-group contract <coop-thread-groups>` as the portable API.
``local`` and ``shared`` expose the active Numba-CUDA-MLIR runtime's memory
namespaces; their allocations follow that compiler's rules.

.. currentmodule:: cuda.coop.numba_mlir

.. autofunction:: ThreadData

.. autoclass:: TempStorage
   :no-members:
   :no-special-members:

.. autoclass:: StatefulFunction
   :no-members:
   :no-special-members:

.. autofunction:: load
.. autofunction:: store
.. autofunction:: reduce
.. autofunction:: sum
.. autofunction:: scan
.. autofunction:: exclusive_sum
.. autofunction:: inclusive_sum
.. autofunction:: exclusive_scan
.. autofunction:: inclusive_scan
.. autofunction:: exchange
.. autofunction:: shuffle

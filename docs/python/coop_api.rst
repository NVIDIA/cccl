.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

.. warning::
   ``cuda.coop`` is an experimental API and is subject to change.

Portable API
------------

The portable functions below are compiler markers. The installed ``.pyi``
files are authoritative for overload and result typing.

.. automodule:: cuda.coop
   :members:
   :exclude-members: __version__
   :imported-members:
   :no-undoc-members:
   :no-special-members:

Numba-CUDA-MLIR-qualified API
-----------------------------

.. py:module:: cuda.coop.numba_mlir

The qualified module provides matching Block, physical-Warp, and logical-Warp
Load, Store, Exchange, and Scan entry points; block Shuffle; hierarchy-aware
Reduce; group descriptors; ``ThreadData``; and ``TempStorage``. It additionally
exposes backend memory namespaces and payload-alignment controls. Portable and
qualified calls use the same lowercase string selectors. Block Load and Store
support ``direct``, ``striped``, ``vectorize``, ``transpose``,
``warp_transpose``, and ``warp_transpose_timesliced``. Physical and logical
Warp calls support ``direct``, ``striped``, ``vectorize``, and ``transpose``.
Use ``this_warp()`` for the physical width of 32 or
``this_warp().group_by(width)`` for a logical width of 1, 2, 4, 8, 16, or 32.
The enclosing block must contain a multiple of 32 threads.

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
per-thread arrays and supports RAKING, RAKING_MEMOIZE, and WARP_SCANS. Warp Scan
accepts one scalar per lane. The qualified API adds stateless operator
callbacks, a one-item ``aggregate_output``, and Warp-only ``valid_items``.
Prefix callback state is not exposed.

For Warp Load and Store, each group receives an automatic memory origin of
``group_index * (group_size * items_per_thread)`` before the caller's element
offset is applied, where the index is the x-major linear thread rank divided by
the group size. Its ``valid_items`` count is relative to that group tile.
``ThreadGroup`` values are descriptor-only; runtime query, membership, and
synchronization methods are not part of this release.

See the :github:`Numba-CUDA-MLIR type declarations
<python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi>` for the complete overload
contract. Importing this qualified module requires the matching
Numba-CUDA-MLIR extra.

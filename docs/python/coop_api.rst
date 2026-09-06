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

The qualified module provides matching Block, physical Warp, and logical Warp
Load, Store, and Exchange entry points plus block Shuffle, group descriptors,
``ThreadData``, and ``TempStorage``. It additionally exposes backend memory
namespaces and payload-alignment controls. Portable and qualified calls use the
same lowercase string selectors. Block Load and Store support ``direct``,
``striped``, ``vectorize``, ``transpose``, ``warp_transpose``, and
``warp_transpose_timesliced``. Physical and logical Warp calls support
``direct``, ``striped``, ``vectorize``, and ``transpose``. Use
``this_warp()`` for the physical width of 32 or
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

Each Warp group receives an automatic memory origin of
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
not a higher level. Mapped warps-within-block groups provide queries and
membership only; their synchronization methods are rejected. Grid
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

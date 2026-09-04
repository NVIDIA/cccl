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

The qualified module provides the matching Block and physical Warp Load and
Store entry points, group descriptors, ``ThreadData``, and ``TempStorage``. It
additionally exposes backend memory namespaces and payload-alignment controls.
Portable and qualified calls use the same lowercase string selectors. Block
calls support ``direct``, ``striped``, ``vectorize``, ``transpose``,
``warp_transpose``, and ``warp_transpose_timesliced``. Physical Warp calls
support ``direct``, ``striped``, ``vectorize``, and ``transpose`` at a fixed
width of 32 threads; the enclosing block must contain a multiple of 32 threads.

``direct``, ``striped``, and ``vectorize`` are storage-free at both scopes.
Physical Warp ``transpose`` uses compiler-owned storage with one disjoint
slice per warp and a ``syncwarp`` reuse barrier. Explicit ``TempStorage`` is
rejected for every
physical Warp algorithm. Transpose Store operations preserve their caller-owned
input payload while CUB performs its internal reordering.

Each physical warp receives an automatic memory origin of
``physical_warp_index * (32 * items_per_thread)`` before the caller's element
offset is applied, where the index is the x-major linear thread rank divided by
32. Its ``valid_items`` count is relative to that warp tile. ``ThreadGroup``
values are descriptor-only. ``group_by`` constructs a static partition
descriptor, but logical-warp Load and Store are deferred; runtime query,
membership, and synchronization methods are not part of this release.

See the :github:`Numba-CUDA-MLIR type declarations
<python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi>` for the complete overload
contract. Importing this qualified module requires the matching
Numba-CUDA-MLIR extra.

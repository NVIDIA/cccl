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
Reduce; group descriptors; ``ThreadData``; ``TempStorage``; and
``StatefulFunction``. It additionally exposes backend memory namespaces and
payload-alignment controls. Portable and qualified calls use the same lowercase
string selectors. Block Load and Store support ``direct``, ``striped``,
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
``ThreadGroup`` values are descriptor-only; runtime query, membership, and
synchronization methods are not part of this release.

See the :github:`Numba-CUDA-MLIR type declarations
<python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi>` for the complete overload
contract. Importing this qualified module requires the matching
Numba-CUDA-MLIR extra.

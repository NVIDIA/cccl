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

The qualified module provides the matching Block Load and Store entry points,
group descriptors, ``ThreadData``, and ``TempStorage``. It additionally exposes
backend memory namespaces and payload-alignment controls. Shared selectors use
the same lowercase strings as the portable API. All six selectors are
executable: ``direct``, ``striped``, ``vectorize``, ``transpose``,
``warp_transpose``, and ``warp_transpose_timesliced``. The first three are
storage-free; the transpose algorithms use CUB temporary storage. Transpose
Store operations preserve their caller-owned input payload while CUB performs
its internal reordering. Enum and integer algorithm selectors are rejected.
``ThreadGroup`` values are descriptor-only.
``group_by`` constructs a static partition descriptor; runtime query,
membership, and synchronization methods are not part of this release.

See the :github:`Numba-CUDA-MLIR type declarations
<python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi>` for the complete overload
contract. Importing this qualified module requires the matching
Numba-CUDA-MLIR extra.

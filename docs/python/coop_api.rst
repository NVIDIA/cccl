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
the complete CUB Block Load and Store algorithm enums, although only DIRECT is
executable in this release. ``ThreadGroup`` values are descriptor-only.
``group_by`` constructs a static partition descriptor; runtime query,
membership, and synchronization methods are not part of this release.

See the :github:`Numba-CUDA-MLIR type declarations
<python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi>` for the complete overload
contract. Importing this qualified module requires the matching
Numba-CUDA-MLIR extra.

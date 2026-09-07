.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

Portable API
------------

Only the portable public surface is rendered here. Runtime functions are
compiler markers, so the installed ``.pyi`` declarations remain authoritative
for overloads and result types.

.. automodule:: cuda.coop
   :members:
   :exclude-members: __version__
   :imported-members:
   :no-undoc-members:
   :no-special-members:

CUTLASS-qualified API
---------------------

.. py:module:: cuda.coop.cutlass

The qualified module provides the same group constructors and primitive
families as the portable API. It additionally exports ``ThreadDataSource``,
``ThreadDataLoadSource``, and ``ThreadDataTensorMetadata`` for adapting CuTe
register values.

See the :github:`CUTLASS type declarations
<python/cuda_coop/cuda/coop/cutlass/__init__.pyi>` for its complete overload
contract. Importing this module requires the CUTLASS extra.

Numba-CUDA-MLIR-qualified API
-----------------------------

.. py:module:: cuda.coop.numba_mlir

The qualified module provides the same group constructors and primitive
families as the portable API. It additionally exports ``local``, ``shared``,
``gpu_dataclass``, ``gpu_dataclass_argument_handler``, ``StatefulFunction``,
and the CUB load, store, scan, and histogram algorithm enums.

See the :github:`Numba-CUDA-MLIR type declarations
<python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi>` for its complete overload
contract. Importing this module requires a Numba-CUDA-MLIR extra.

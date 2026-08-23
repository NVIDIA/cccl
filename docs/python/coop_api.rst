.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

Portable API
------------

.. automodule:: cuda.coop
   :members:
   :imported-members:

Qualified backend APIs
----------------------

The backend packages validate their optional compiler runtimes when imported,
so their surfaces are described here without importing them into the
documentation build. Both mirror the portable root API: every group-first
operation listed above is exported with the same signature and contract, and
each package re-exports ``ThreadData``, ``TempStorage``, ``ThreadGroup``,
``Hierarchy``, and the ``this_*`` group constructors. The wheel's ``.pyi``
stubs are the authoritative typed declaration of each qualified surface.

.. py:module:: cuda.coop.cutlass

``cuda.coop.cutlass`` activates the CUTLASS CuTe DSL backend explicitly. On
import it validates the installed ``nvidia-cutlass-dsl`` runtime, registers a
compiler trace context, and becomes the common-root fallback backend. Beyond
the portable operations it adds deferred ``make_*`` operation factories and
``Payload`` selection for CUTLASS array ("Prims") payloads. See
:doc:`coop_cutlass`.

.. py:module:: cuda.coop.numba_mlir

``cuda.coop.numba_mlir`` activates the Numba-CUDA-MLIR backend explicitly. On
import it verifies the installed ``numba-cuda-mlir`` compiler hooks and
transactionally registers the cooperative whole-function rewrites. Beyond the
portable operations it adds scoped ``_block``/``_warp`` factory surfaces,
``gpu_dataclass`` temp-storage traits, shared-memory ``TempStorage`` planning,
and runtime-free algorithm enums. See :doc:`coop_numba_mlir`.

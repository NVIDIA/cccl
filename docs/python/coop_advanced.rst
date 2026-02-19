.. _cccl-python-coop-advanced:

Advanced Topics
===============

Batched LTOIR (default)
-----------------------

``cuda.coop`` batches LTO-IR compilation for all primitives in a kernel by
default. This reduces NVRTC overhead when multiple primitives are used in a
single kernel.

To disable bundling (debugging or performance experiments):

.. code-block:: bash

   export NUMBA_CCCL_COOP_BUNDLE_LTOIR=0

Debugging and instrumentation
-----------------------------

Environment variables that are useful during development:

.. list-table::
   :header-rows: 1

   * - Variable
     - Purpose
   * - ``NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT=1``
     - Enable an NVRTC compile counter.
   * - ``NUMBA_CCCL_COOP_NVRTC_DUMP=1``
     - Dump NVRTC input sources to ``/tmp/cccl_nvrtc``.
   * - ``NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/path``
     - Override the dump directory.
   * - ``NUMBA_CCCL_COOP_DEBUG=1``
     - Enable cuda.coop debug logging.
   * - ``NUMBA_CCCL_COOP_INJECT_PRINTFS=1``
     - Inject device printfs (debug only).
   * - ``CCCL_ENABLE_CACHE=1``
     - Enable disk caching for NVRTC outputs.

If you see multiple NVRTC compiles for a single kernel, ensure bundling is
enabled and check the dump output to confirm which shims are being generated.

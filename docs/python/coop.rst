.. _cccl-python-coop:

``cuda.coop``: Cooperative GPU Primitives
==========================================

``cuda.coop`` brings CCCL cooperative primitives to Python kernel DSLs. A
kernel names the participating CUDA thread group, describes each thread's
register values with :func:`~cuda.coop.ThreadData`, and passes both to a
group-first collective.

Portable root API
-----------------

The root API contains the behavior shared by CUTLASS Python DSL and
Numba-CUDA-MLIR. This Numba-CUDA-MLIR kernel loads two items per thread,
computes a block-wide exclusive sum, and stores the result:

.. literalinclude:: ../../python/cuda_coop/examples/numba_mlir/common_block_scan.py
   :language: python
   :start-after: # docs: start numba-common-block-scan
   :end-before: # docs: end numba-common-block-scan

The CUTLASS version has the same kernel body. Only its decorator, tensor
annotations, and launch syntax change. Both complete programs are linked in
the :ref:`cuda-coop-examples` section below.

Backend discovery
-----------------

Importing :mod:`cuda.coop` probes an explicit allowlist containing CUTLASS
Python DSL and Numba-CUDA-MLIR. ``cuda.coop`` checks the compiler hooks needed
by each installed adapter before registering it. Missing runtimes are ignored.
An incompatible installed runtime emits ``CudaCoopAutoRegistrationWarning``;
another compatible backend can still register.

Set ``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1`` before import to skip these
probes. Applications that manage compiler activation can then import one
qualified backend explicitly:

.. code-block:: python

   import cuda.coop.cutlass as coop
   # or
   import cuda.coop.numba_mlir as coop

The base import succeeds when neither compiler is installed. Cooperative
operations still require a compatible compilation context.

Groups, payloads, and scratch
-----------------------------

Every operation receives a :class:`~cuda.coop.ThreadGroup` first. Physical
thread, warp, block, cluster, and grid descriptors come from the current
launch. ``group_by()`` partitions a warp into smaller logical groups where an
operation supports them. All selected members must reach a collective.

``ThreadData(items_per_thread, dtype=...)`` describes one fixed-size register
payload per thread. Most operations return a fresh payload and leave the input
unchanged. ``TempStorage()`` requests compiler-planned scratch space;
``TempStorage(size_in_bytes, alignment=...)`` supplies an explicit capacity.

The portable operation and group combinations are:

.. list-table:: Portable operations
   :header-rows: 1
   :widths: 18 52 30

   * - Family
     - Calls
     - Groups
   * - Data movement
     - ``load``, ``store``, ``exchange``
     - block, warp, logical warp
   * - Reduction
     - ``reduce``, ``sum``
     - thread, block, warp, logical warp, cluster
   * - Scan
     - ``scan``, ``exclusive_sum``, ``inclusive_sum``,
       ``exclusive_scan``, ``inclusive_scan``
     - block, warp, logical warp
   * - Neighbors
     - ``adjacent_difference``, ``discontinuity``, ``shuffle``
     - block
   * - Comparison sort
     - ``merge_sort_keys``, ``merge_sort_pairs``
     - block, warp, logical warp
   * - Radix
     - ``radix_sort_keys``, ``radix_sort_pairs``, ``radix_rank``
     - block
   * - Counting
     - ``histogram``, ``run_length_decode``
     - block
   * - Selection
     - ``topk_min_keys``, ``topk_min_pairs``, ``topk_max_keys``,
       ``topk_max_pairs``
     - block

Operations reject unsupported group kinds before creating compiler artifacts.
``valid_items`` denotes a uniform flattened prefix. TopK defines only the
first ``k`` flattened blocked positions and does not sort that prefix.

Qualified backend controls
--------------------------

The root API validates group kind, dtype, payload shape, static controls, and
result layout before dispatch. Qualified modules retain backend-specific dtype
support, payload adapters, callbacks, algorithms, and optional outputs.

This CUTLASS fragment selects a stable radix ordering and an unordered TopK
prefix. The fixed scratch descriptor uses the public ``alignment`` spelling:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/qualified_radix_topk.py
   :language: python
   :start-after: # docs: start cutlass-qualified-ordering
   :end-before: # docs: end cutlass-qualified-ordering
   :dedent: 4

Numba-CUDA-MLIR can request the relative offset within each decoded run and the
total decoded stream size:

.. literalinclude:: ../../python/cuda_coop/examples/numba_mlir/qualified_histogram_decode.py
   :language: python
   :start-after: # docs: start numba-qualified-run-length-decode
   :end-before: # docs: end numba-qualified-run-length-decode
   :dedent: 4

.. _cuda-coop-examples:

Examples
--------

The complete examples allocate device inputs, launch one block, and verify the
result on the host:

* :download:`portable CUTLASS block scan
  <../../python/cuda_coop/examples/cutlass/common_block_scan.py>`
* :download:`qualified CUTLASS radix sort and TopK
  <../../python/cuda_coop/examples/cutlass/qualified_radix_topk.py>`
* :download:`portable Numba-CUDA-MLIR block scan
  <../../python/cuda_coop/examples/numba_mlir/common_block_scan.py>`
* :download:`qualified Numba-CUDA-MLIR histogram and run-length decode
  <../../python/cuda_coop/examples/numba_mlir/qualified_histogram_decode.py>`

API reference
-------------

See :doc:`coop_api` for the exact portable signatures and result contracts.

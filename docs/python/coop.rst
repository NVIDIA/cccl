.. _cccl-python-coop:

``cuda.coop``: Cooperative GPU Primitives
==========================================

``cuda.coop`` brings CCCL cooperative primitives to Python kernel DSLs. A
kernel names the participating CUDA thread group, describes each thread's
register values with :func:`~cuda.coop.ThreadData`, and passes both to a
group-first collective.

Installation
------------

Install the extra for the compiler and CUDA major version in the target
environment:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

   python -m pip install "cuda-coop[cutlass]"

The base ``cuda-coop`` distribution contains the portable API, type
declarations, and CCCL headers. The Numba-CUDA-MLIR extra requires its public
0.5 runtime. The CUTLASS extra requires ``nvidia-cutlass-dsl>=4.8`` and CUDA
13; it cannot resolve from an index that does not yet carry that DSL release.

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
launch. :meth:`~cuda.coop.ThreadGroup.group_by` partitions a warp into smaller
logical groups or a block into a group of warps. The table below is
authoritative: mapped-block groups are backend-qualified, and no portable
collective currently accepts the grid descriptor. All selected members must
reach a collective.

``ThreadData(items_per_thread, dtype=...)`` describes one fixed-size register
payload per thread. Most operations return a fresh payload and leave the input
unchanged. ``TempStorage()`` requests compiler-planned scratch space;
``TempStorage(size_in_bytes, alignment=...)`` supplies an explicit capacity.
The builtin ``int`` and ``float`` dtype tokens mean 32-bit integer and
floating-point values; use an explicit NumPy or compiler dtype token for
another width.

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

The optional ``valid_items`` count identifies the valid prefix of a partially
filled group tile. ``load`` fills tail positions with the ``oob_default``
sentinel. ``store`` can use the same count to leave the destination tail
untouched. Both operations accept an element ``offset`` into the source or
destination; it is not a byte offset. ``oob_default`` requires ``valid_items``;
the element offset is independent. All three controls are keyword-only.
The portable load/store surface accepts block, physical-warp, and logical-warp
groups. Temporary storage and the ``warp_transpose`` and
``warp_transpose_timesliced`` algorithms are block-only; unsupported groups are
rejected before the backend creates compiler artifacts.
TopK defines only the first ``k`` flattened blocked positions and does not
sort that prefix.

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

The backend differences are deliberate compiler integrations, not differences
in the portable semantics:

.. list-table:: Qualified backend differences
   :header-rows: 1
   :widths: 22 39 39

   * - Concern
     - CUTLASS Python DSL
     - Numba-CUDA-MLIR
   * - Register payloads
     - Adapts CuTe register tensors and exposes source metadata protocols.
     - Allocates compiler ``local``/``shared`` values and supports registered
       GPU dataclasses.
   * - Algorithm controls
     - Uses portable algorithm tokens and trace-time CUTLASS values.
     - Also exposes CUB block/warp algorithm enums and device callables.
   * - Block-scan prefixes
     - No qualified prefix-callback API.
     - Accepts a stateless ``prefix_op`` or a ``StatefulFunction`` with an
       explicit one-item running state. Callbacks are block-only.
   * - Compilation
     - Records providers while tracing and links them at trace finalization.
     - Plans before type inference and materializes providers after inference.

For a stateless Numba block-scan prefix, pass a device callable that maps the
block aggregate to the prefix. For a stateful prefix, pair the callback with
its state dtype and pass a one-item ``ThreadData`` as the third positional
argument:

.. code-block:: python

   from numba_cuda_mlir import types
   import cuda.coop.numba_mlir as coop

   class RunningPrefix:
       def __call__(self_ptr, block_aggregate):
           previous = self_ptr[0]
           self_ptr[0] = previous + block_aggregate
           return previous

   running_prefix = coop.StatefulFunction(RunningPrefix, types.int32)

   # Inside a kernel; every block member reaches each scan call.
   prefix_state = coop.ThreadData(1, dtype=types.int32)
   prefix_state[0] = 0

   items = coop.ThreadData(1, dtype=types.int32)
   items[0] = 1
   scanned = coop.exclusive_sum(
       coop.this_block(), items, prefix_state, prefix_op=running_prefix
   )

Reuse ``prefix_state`` without reinitializing it in later block-tile scan calls
to carry the running prefix forward.

``block_prefix_callback_op`` remains an accepted compatibility spelling for
``prefix_op``. Warp scans intentionally do not accept either callback form.

CUTLASS AOT packs
-----------------

The CUTLASS backend can capture exact provider LTO-IR bundles with
``cuda-coop-aot`` or :func:`cuda.coop.cutlass.aot.capture` and select them with
:func:`cuda.coop.cutlass.aot.use`. Treat a pack as executable device code:
digests detect corruption but do not authenticate its producer or establish
that its LTO-IR is safe. Consume only packs from a build or producer you trust.

Reuse matches the provider ABI, rendered source, bundle format, target
architecture, compiler options, layout expressions, and nvJitLink
compatibility. The writer version is informational. Exact hits intentionally
avoid current-header discovery, so a header or provider change that alters the
ABI or semantics of otherwise identical rendered source requires a
provider-ABI bump.

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

See :doc:`coop_api` for runtime signatures and summary documentation. The
installed ``.pyi`` files provide the authoritative static overload and result
typing contracts, including backend-qualified controls.

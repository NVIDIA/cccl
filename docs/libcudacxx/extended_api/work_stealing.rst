.. _libcudacxx-extended-api-work-stealing:

Work stealing
=============

Defined in header ``<cuda/work_stealing>`` if the CUDA compiler supports at least PTX ISA 8.7:

.. code:: cuda

   namespace cuda {

       template <int ThreadBlockRank = 3, invocable<dim3> UnaryFunction = ..unspecified..>
       __device__ void for_each_canceled_block(UnaryFunction uf);

       template <int ThreadBlockRank = 3, invocable<dim3> UnaryFunction = ..unspecified..>
       __device__ void for_each_canceled_cluster(UnaryFunction uf);

   } // namespace cuda

**Note**: On devices with compute capability 10.0 or higher, this function may leverage hardware acceleration.

These APIs are primarily intended for implementing work-stealing at the thread-block or cluster level.

Compared to alternative work distribution techniques, such as  `grid-stride loops <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>`__, which distribute work statically, or dynamic work distribution methods relying on global memory concurrency, these API offer several advantages:

   - **Dynamic work-stealing**, i.e., thread blocks that complete their tasks sooner may take on additional work from slower thread blocks.
   - **GPU Work Scheduler cooperation**, e.g., to respect work priorities and improve load balancing.
   - **Lower latency**, e.g., when compared to global memory atomics.

For better performance, extract the shared prologue and epilogue from the work to be performed to reuse them across iterations:

  - Prologue: Initialization code and data common to all thread blocks or clusters, such as ``__shared__`` memory allocation and initialization.
  - Epilogue: Finalization code common to all thread blocks or clusters, such as writing shared memory back to global memory.

The ``for_each_canceled_cluster`` API may be used with thread-block clusters of any size, including one.
The ```for_each_canceled_block`` API is optimized for and requires thread-block clusters of size one.

**Mandates**:

   - ``ThreadBlockRank`` equals the rank of the thread block: ``1``, ``2``, or ``3`` for one-dimensional, two-dimensional, and three-dimensional thread blocks, respectively.
   - ``is_invokable_r_v<UnaryFunction, void, dim3>`` is true.

**Preconditions**:

   - ``for_each_canceled_block`` shall only be called from grids with **exactly** one thread block per cluster.
   - All threads within a thread-block cluster shall call either ``for_each_canceled_block`` or ``for_each_canceled_cluster``, and do so **exactly once**.

**Effects**:

   - Invokes ``uf`` with ``blockIdx`` and then repeatedly attempts to cancel the launch of another thread block or cluster within the current grid:

      - If successful: invokes ``uf`` with the canceled thread block's ``blockIdx`` and repeats.
      - Otherwise, the function returns; it failed to cancel the launch of another thread block or cluster.

**Remarks**: ``for_each_canceled_cluster` guarantees that the relative position within a cluster of the thread block index ``uf`` is invoked with is always the same.

Example
-------

This example demonstrates work-stealing at thread-block granularity using this API.

.. literalinclude:: ../libcudacxx/examples/work_stealing.cu
    :language: c++

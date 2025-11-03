.. _device-module:

Device-Wide Primitives
==================================================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   ../api/device


Determining Temporary Storage Requirements
++++++++++++++++++++++++++++++++++++++++++++++++++

**Two-Phase API** (Traditional)

Most CUB device-wide algorithms follow a two-phase usage pattern:

1. **Query Phase**: Call the algorithm with ``d_temp_storage = nullptr`` to determine required temporary storage size
2. **Execution Phase**: Allocate storage and call the algorithm again to perform the actual operation

**What arguments are needed during the query phase?**

* **Required**: Data types (via template parameters and iterator types) and problem size (``num_items``)
* **Can be ``nullptr``/dummy**: All device pointers (``d_in``, ``d_out``, etc.).
* **Why this is safe**: During the size query (``d_temp_storage == nullptr``) our dispatch layer exits immediately—no kernels launch and user-provided pointers are never dereferenced. Only the ``temp_storage_bytes`` reference is written.

Example pattern:

.. literalinclude:: ../../../cub/examples/device/example_device_reduce.cu
   :language: c++
   :dedent:
   :start-after: example-begin temp-storage-query
   :end-before: example-end temp-storage-query

**Single-Phase API** (Environment-Based)

Environment-based overloads are being rolled out across CUB's device-wide primitives (rollout in progress).
They eliminate explicit temporary-storage management, which in turn removes:

- the two-phase query/execute call sequence, and
- the two legacy storage arguments at the beginning of each API.

Key properties of the execution environment argument:

- It is the last parameter and is defaulted (you can omit it entirely).
- You can specify the CUDA stream via the environment.
- You can select the memory resource (CCCL-provided or a custom resource) that backs internal allocations.
- For some algorithms, you can request determinism requirements (e.g., gpu-to-gpu) via the environment.
- Multiple properties can be provided simultaneously in a single, centralized argument.

Example (centralized control via a single environment argument):

.. code-block:: c++

   #include <cub/cub.cuh>
   #include <cuda/std/execution>
   #include <cuda/stream_ref>
   #include <cuda/__memory_resource/get_memory_resource.h>
   #include <cuda/__execution/determinism.h>

   // Build an execution environment with stream, memory resource, and determinism
   cudaStream_t stream = /* ... */;
   auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};

   auto mr = /* CCCL-provided or user-defined device_memory_resource */;
   auto mr_env = cuda::std::execution::prop{cuda::mr::__get_memory_resource_t{}, mr};

   auto det_env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);

   auto env = cuda::std::execution::env{stream_env, mr_env, det_env};

   // Single-phase API (no explicit temp storage, environment last and defaulted)
   cub::DeviceReduce::Reduce(d_in, d_out, num_items, cuda::std::plus<>{}, init, env);

This page focuses on the traditional two-phase pattern; see individual algorithm documentation for the
availability and specifics of single-phase overloads.

CUB device-level single-problem parallel algorithms:

* :cpp:struct:`cub::DeviceAdjacentDifference` computes the difference between adjacent elements residing within device-accessible memory
* :cpp:struct:`cub::DeviceFor` provides device-wide, parallel operations for iterating over data residing within device-accessible memory
* :cpp:struct:`cub::DeviceHistogram` constructs histograms from data samples residing within device-accessible memory
* :cpp:struct:`cub::DevicePartition` partitions data residing within device-accessible memory
* :cpp:struct:`cub::DeviceMerge` merges two sorted sequences in device-accessible memory into a single one
* :cpp:struct:`cub::DeviceMergeSort` sorts items residing within device-accessible memory
* :cpp:struct:`cub::DeviceRadixSort` sorts items residing within device-accessible memory using radix sorting method
* :cpp:struct:`cub::DeviceReduce` computes reduction of items residing within device-accessible memory
* :cpp:struct:`cub::DeviceRunLengthEncode` demarcating "runs" of same-valued items within a sequence residing within device-accessible memory
* :cpp:struct:`cub::DeviceScan` computes a prefix scan across a sequence of data items residing within device-accessible memory
* :cpp:struct:`cub::DeviceSelect` compacts data residing within device-accessible memory


CUB device-level segmented-problem (batched) parallel algorithms:

* :cpp:struct:`cub::DeviceSegmentedSort` computes batched sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceSegmentedRadixSort` computes batched radix sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceSegmentedReduce` computes reductions across multiple sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceCopy` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceMemcpy` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory

.. _device-module:

Device-Wide Primitives
======================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   ../api/device

Almost all of CUB's device-wide APIs come in two flavors:

* the traditional two-phase style that requires calling the API twice and managing temporary storage explicitly,
* and the newer single-phase style where temporary storage is obtained from a memory resource in the execution environment.

Some APIs that do not require any temporary storage may have a traditional single-phase overload in addition to an newer environment one.

.. _device-temp-storage:

Two-Phase API (explicit temporary storage management)
+++++++++++++++++++++++++++++++++++++++++++++++++++++

Traditional two-phase APIs can be recognized by taking ``void* d_temp_storage, size_t& temp_storage_bytes`` as their first two parameters.
They follow a two-phase usage pattern that requires three steps:

1. **Query Phase**: The algorithm is called the first time with ``d_temp_storage = nullptr`` to determine the required temporary storage size.
   The needed size in bytes is written to the parameter ``temp_storage_bytes``.
2. **Temporary storage allocation**: The user is responsible to allocate device memory of at least ``temp_storage_bytes`` bytes.
3. **Execution Phase**: The algorithm is called the second time with ``d_temp_storage`` pointing to the allocated device memory, performing the actual operation.

In principle, the query phase and execution phase must call the same CUB API.
This means in detail:

* **Template arguments**: The query call must use the same template arguments as the execution call, so they share the same template instantiation.
* **Argument values**: Regarding function parameters, only the values of the ``d_temp_storage``, ``temp_storage_bytes``,
  and problem-size related arguments (like number of elements, number of segments, segment sizes, etc.) may be read during the query phase.
  No other parameters (like input/output iterators, initial values, etc.) are accessed during the query phase, so their values may be indeterminate.
  During the query phase, the API will return before launching any kernels or touching user storage.
* **Current device**: The computed temporary storage size is valid only when the execution phase runs on the same current CUDA device as the query.
  Re-run the query if the current device changes between phases.

Example pattern:

.. literalinclude:: ../../../cub/examples/device/example_device_reduce.cu
   :language: c++
   :dedent:
   :start-after: example-begin temp-storage-query
   :end-before: example-end temp-storage-query


Environment API (single phase)
++++++++++++++++++++++++++++++

Environment-based overloads are available for all CUB device-wide algorithms.
They remove the split of query/execute phase and manually obtaining the temporary storage.
Instead, the temporary storage is automatically requested from a memory resource queried from the execution environment argument.
The environment supports further properties like passing a stream or an execution requirement in addition to a memory resource.

Key properties of the environment argument:

- It is a defaulted parameter and appears as the last argument.
- Streams like `cudaStream_t` or `cuda::stream_ref` can be passed as environments directly, or added to the environment.
- You can select the memory resource (CCCL-provided or custom) used for internal allocations.
- Supported algorithms accept determinism requirements (for example, ``cuda::execution::determinism::gpu_to_gpu``).
- Multiple properties compose into a single centralized argument by wrapping them into a ``cuda::execution::env`` object.

Example pattern:

.. literalinclude:: ../../../cub/examples/device/example_device_reduce_env.cu
   :language: c++
   :dedent:
   :start-after: example-begin env-overload-setup
   :end-before: example-end env-overload-setup

.. literalinclude:: ../../../cub/examples/device/example_device_reduce_env.cu
   :language: c++
   :dedent:
   :start-after: example-begin env-overload-run
   :end-before: example-end env-overload-run


API overview
++++++++++++

In the following, the various groups of CUB device-wide algorithms are listed,
linking to their respective documentation.

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
* :cpp:struct:`cub::DeviceTransform` transforms elements from multiple input sequences into an output sequence
* :cpp:struct:`cub::DeviceTopK` finds the largest (or smallest) K items from an unordered list residing within device-accessible memory


CUB device-level segmented-problem (batched) parallel algorithms:

* :cpp:struct:`cub::DeviceSegmentedSort` computes batched sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceSegmentedRadixSort` computes batched radix sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceSegmentedReduce` computes reductions across multiple sequences of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceCopy` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceMemcpy` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory
* :cpp:struct:`cub::DeviceFind` provides vectorized binary search algorithms

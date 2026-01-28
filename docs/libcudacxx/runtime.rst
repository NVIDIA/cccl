.. _libcudacxx-runtime-api:

Runtime
=======

.. toctree::
   :hidden:
   :maxdepth: 1

   runtime/stream
   runtime/event
   runtime/algorithm
   runtime/device
   runtime/hierarchy
   runtime/launch
   runtime/buffer
   runtime/memory_resource
   runtime/memory_pools

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **API**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`devices <cccl-runtime-device-devices>`
     - A range of all available CUDA devices
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`device_ref <cccl-runtime-device-device-ref>`
     - A non-owning representation of a CUDA device
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`arch_traits <cccl-runtime-device-arch-traits>`
     - Per-architecture trait accessors
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`stream_ref <cccl-runtime-stream-stream-ref>`
     - A non-owning wrapper around a ``cudaStream_t``
     - CCCL 2.2.0
     - CUDA 12.3

   * - :ref:`stream <cccl-runtime-stream-stream>`
     - An owning wrapper around a ``cudaStream_t``
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`event_ref <cccl-runtime-event-event-ref>`
     - A non-owning wrapper around a ``cudaEvent_t``
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`event <cccl-runtime-event-event>`
     - An owning wrapper around a ``cudaEvent_t`` (timing disabled)
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`timed_event <cccl-runtime-event-timed-event>`
     - An owning wrapper around a ``cudaEvent_t`` with timing enabled and elapsed-time queries
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`copy_bytes <cccl-runtime-algorithm-copy_bytes>`
     - Byte-wise copy into a ``cuda::stream_ref`` for ``cuda::std::span``/``cuda::std::mdspan`` sources and destinations
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`fill_bytes <cccl-runtime-algorithm-fill_bytes>`
     - Byte-wise fill into a ``cuda::stream_ref`` for ``cuda::std::span``/``cuda::std::mdspan`` destinations
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`hierarchy <cccl-runtime-hierarchy-hierarchy>`
     - Representation of CUDA thread hierarchies (grid, cluster, block, warp, thread)
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`launch <cccl-runtime-launch-launch>`
     - Kernel launch with configuration and options
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`kernel_config <cccl-runtime-launch-kernel-config>`
     - Kernel launch configuration combining hierarchy dimensions and launch options
     - CCCL 3.2.0
     - CUDA 13.2
     
   * - :ref:`make_config <cccl-runtime-launch-make-config>`
     - Factory function to create kernel configurations from hierarchy dimensions and launch options
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`device_memory_pool <cccl-runtime-memory-pools-device-memory-pool>`
     - Stream-ordered device memory pool using CUDA memory pool API
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`managed_memory_pool <cccl-runtime-memory-pools-managed-memory-pool>`
     - Stream-ordered managed (unified) memory pool
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`pinned_memory_pool <cccl-runtime-memory-pools-pinned-memory-pool>`
     - Stream-ordered pinned (page-locked) host memory pool
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`device_default_memory_pool <cccl-runtime-memory-pools-device-default>`
     - Get the default device memory pool for a device
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`managed_default_memory_pool <cccl-runtime-memory-pools-managed-default>`
     - Get the default managed (unified) memory pool
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`pinned_default_memory_pool <cccl-runtime-memory-pools-pinned-default>`
     - Get the default pinned (page-locked) host memory pool
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`Memory Resources (Extended API) <libcudacxx-extended-api-memory-resources>`
     - ``cuda::mr`` concepts, properties, and resource implementations
     - CCCL 2.2.0 (experimental), CCCL 3.1.0 (stable)
     - CUDA 12.3 (experimental), CUDA 13.1 (stable)

   * - :ref:`buffer <cccl-runtime-buffer-buffer>`
     - Typed data container allocated from memory resources. It handles stream-ordered allocation, initialization, and deallocation of memory.
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`any_resource <cccl-runtime-memory-resource-any-resource>`
     - Type-erased wrapper that owns any memory resource satisfying specified properties
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`resource_ref <cccl-runtime-memory-resource-resource-ref>`
     - Non-owning, type-erased wrapper around a stream-ordered memory resource
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`shared_resource <cccl-runtime-memory-resource-shared-resource>`
     - Reference-counted wrapper for sharing memory resources
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`synchronous_resource_adapter <cccl-runtime-memory-resource-synchronous-adapter>`
     - Adapter that enables synchronous resources to work with streams
     - CCCL 3.2.0
     - CUDA 13.2

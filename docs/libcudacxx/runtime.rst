.. _libcudacxx-runtime-api:

Runtime
=======

The Runtime API provides higher-level building blocks for core CUDA functionality. It takes the existing CUDA Runtime API
set and removes or replaces some problematic patterns, such as implicit state. It is designed to make common operations
like resource management, work submission, and memory allocation easier to express and safer to compose. These APIs lower
to the CUDA Driver API under the hood, but remain composable with the CUDA Runtime API by reusing runtime handle types
(such as ``cudaStream_t``) in the interfaces. This results in an interface that applies RAII for lifetime management,
while remaining composable with existing CUDA C++ code that manages resources explicitly.

At a glance, the runtime layer includes:

- Streams and events work submission and synchronization.
- Buffers as a typed, stream-ordered storage with property-checked memory container.
- Memory pools to allocate device, managed, and pinned memory, either directly or through buffers.
- Launch API to configure and launch kernels.
- Runtime algorithms like ``copy_bytes`` and ``fill_bytes`` for basic data movement.
- Legacy memory resources as synchronous compatibility fallbacks for older toolkits.

See :ref:`CUDA Runtime interactions <cccl-runtime-cudart-interactions>` if you are interested in CUDA Runtime interop.

Example: vector add with buffers, pools, and launch
---------------------------------------------------

.. code:: cpp

   #include <cuda/devices>
   #include <cuda/stream>
   #include <cuda/std/span>
   #include <cuda/buffer>
   #include <cuda/memory_pool>
   #include <cuda/launch>

   struct kernel {
     template <typename Config>
     __device__ void operator()(Config config,
                                cuda::std::span<const float> A,
                                cuda::std::span<const float> B,
                                cuda::std::span<float> C) {
       auto tid = cuda::gpu_thread.rank(cuda::grid, config);
       if (tid < A.size())
         C[tid] = A[tid] + B[tid];
     }
   };

   int main() {
     cuda::device_ref device = cuda::devices[0];
     cuda::stream stream{device};
     auto pool = cuda::device_default_memory_pool(device);

     int num_elements = 1000;
     auto A = cuda::make_buffer<float>(stream, pool, num_elements, 1.0);
     auto B = cuda::make_buffer<float>(stream, pool, num_elements, 2.0);
     auto C = cuda::make_buffer<float>(stream, pool, num_elements, cuda::no_init);

     constexpr int threads_per_block = 256;
     auto config = cuda::distribute<threads_per_block>(num_elements);

     cuda::launch(stream, config, kernel{}, A, B, C);
   }

.. toctree::
   :hidden:
   :maxdepth: 1

   runtime/cudart_interactions
   runtime/stream
   runtime/event
   runtime/algorithm
   runtime/device
   runtime/hierarchy
   runtime/launch
   runtime/buffer
   runtime/memory_pools
   runtime/legacy_resources

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

   * - :ref:`buffer <cccl-runtime-buffer-buffer>`
     - Typed data container allocated from memory resources. It handles stream-ordered allocation, initialization, and deallocation of memory.
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`legacy resources <cccl-runtime-legacy-resources>`
     - Synchronous compatibility resources backed by legacy CUDA allocation APIs.
     - CCCL 3.2.0
     - CUDA 13.2

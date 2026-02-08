.. _cccl-runtime-memory-pools:

Memory Pools
============

Memory pools provide efficient, stream-ordered memory allocation using CUDA's memory pool API. They support both synchronous and stream-ordered allocation/deallocation and can be configured with various memory spaces, properties and attributes.

Memory pool objects implement the :ref:`cuda::memory_resource <libcudacxx-extended-api-memory-resources-resource>` interface with ``allocate(stream, size, alignment)`` and ``deallocate(stream, ptr, size, alignment)`` member functions. They also provide synchronous variants with ``allocate_sync(size, alignment)`` and ``deallocate_sync(ptr, size, alignment)`` member functions. For all of them, the alignment argument is optional.

For the full memory resource model and property system, see :ref:`Memory Resources (Extended API) <libcudacxx-extended-api-memory-resources>`.

Host memory pools are supported on CUDA 12.6 and later. Managed memory pools are supported on CUDA 13.0 and later and are not supported on Windows. For those cases use :ref:`cuda::mr::legacy_pinned_memory_resource <libcudacxx-memory-resource-legacy-pinned-memory-resource>` and :ref:`cuda::mr::legacy_managed_memory_resource <libcudacxx-memory-resource-legacy-managed-memory-resource>` instead.

``cuda::device_memory_pool``
----------------------------
.. _cccl-runtime-memory-pools-device-memory-pool:

``cuda::device_memory_pool`` allocates device memory using CUDA's stream-ordered memory pool API (``cudaMallocFromPoolAsync`` / ``cudaFreeAsync``). When constructed, it creates and owns an underlying ``cudaMemPool_t`` with location type set to ``cudaMemLocationTypeDevice``.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>
   #include <cuda/devices>

   void use_device_pool(cuda::stream_ref stream) {
     // Create a device memory pool
     cuda::device_memory_pool pool{cuda::devices[0]};

     // Allocate memory in stream order
     void* ptr = pool.allocate(stream, 1024, 16);

     // Use memory...

     // Deallocate in stream order
     pool.deallocate(stream, ptr, 1024, 16);
   }

``cuda::device_memory_pool_ref``
---------------------------------
.. _cccl-runtime-memory-pools-device-memory-pool-ref:

``cuda::device_memory_pool_ref`` is a non-owning reference to a device memory pool. It does not own the underlying ``cudaMemPool_t``, so the user must ensure the pool's lifetime exceeds the reference's lifetime.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_pool_ref(cuda::stream_ref stream, cuda::device_memory_pool_ref pool_ref) {
     void* ptr = pool_ref.allocate(stream, 1024);
     // Use memory...
     pool_ref.deallocate(stream, ptr, 1024);
   }

``cuda::managed_memory_pool``
-----------------------------
.. _cccl-runtime-memory-pools-managed-memory-pool:

``cuda::managed_memory_pool`` allocates managed (unified) memory using CUDA's memory pool API. It creates and owns an underlying ``cudaMemPool_t`` with allocation type set to ``cudaMemAllocationTypeManaged``. Managed memory is accessible from both host and device.

Availability: CCCL 3.2.0 / CUDA 13.2 (requires CTK 13.0+). Not supported on Windows

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_managed_pool(cuda::stream_ref stream) {
     cuda::managed_memory_pool pool{};

     // Allocate managed memory
     void* ptr = pool.allocate(stream, 1024);

     // Accessible from both host and device
     // Use memory...

     pool.deallocate(stream, ptr, 1024);
   }

``cuda::managed_memory_pool_ref``
----------------------------------
.. _cccl-runtime-memory-pools-managed-memory-pool-ref:

``cuda::managed_memory_pool_ref`` is a non-owning reference to a managed memory pool.

Availability: CCCL 3.2.0 / CUDA 13.2 (requires CTK 13.0+). Not supported on Windows

``cuda::pinned_memory_pool``
-----------------------------
.. _cccl-runtime-memory-pools-pinned-memory-pool:

``cuda::pinned_memory_pool`` allocates pinned (page-locked) host memory using CUDA's memory pool API. Pinned memory enables faster host-to-device transfers and can be accessed from all devices. The pool can be optionally created for a specific host NUMA node.

Availability: CCCL 3.2.0 / CUDA 13.2 (requires CTK 12.6+)

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_pinned_pool(cuda::stream_ref stream) {
     // Create pinned memory pool
     cuda::pinned_memory_pool pool{};

     // Allocate pinned memory
     void* ptr = pool.allocate(stream, 1024);

     // Use for fast host-device transfers...

     pool.deallocate(stream, ptr, 1024);
   }

   // With NUMA node
   void use_pinned_pool_numa(cuda::stream_ref stream, int numa_id) {
     cuda::pinned_memory_pool pool{numa_id};
     void* ptr = pool.allocate(stream, 1024);
     // Use memory...
     pool.deallocate(stream, ptr, 1024);
   }

``cuda::pinned_memory_pool_ref``
---------------------------------
.. _cccl-runtime-memory-pools-pinned-memory-pool-ref:

``cuda::pinned_memory_pool_ref`` is a non-owning reference to a pinned memory pool.

Availability: CCCL 3.2.0 / CUDA 13.2 (requires CTK 12.6+)

Default Memory Pools
--------------------
.. _cccl-runtime-memory-pools-default-pools:

CUDA provides default memory pools for each memory type. These pools are managed by the CUDA runtime and can be accessed through helper functions. Default pools are useful when you don't need custom pool configuration and want to use the system defaults.

``cuda::device_default_memory_pool``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _cccl-runtime-memory-pools-device-default:

``cuda::device_default_memory_pool(device_ref)`` returns a non-owning reference to the default device memory pool for the specified device. The default pool is created automatically by CUDA and is shared across all users of the device.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/devices>
   #include <cuda/stream>

   void use_default_device_pool(cuda::stream_ref stream) {
     // Get the default device memory pool
     auto pool = cuda::device_default_memory_pool(cuda::devices[0]);

     // Allocate from the default pool
     void* ptr = pool.allocate(stream, 1024);

     // Use memory...

     // Deallocate back to the pool
     pool.deallocate(stream, ptr, 1024);
   }

``cuda::managed_default_memory_pool``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _cccl-runtime-memory-pools-managed-default:

``cuda::managed_default_memory_pool()`` returns a non-owning reference to the default managed (unified) memory pool.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_default_managed_pool(cuda::stream_ref stream) {
     // Get the default managed memory pool
     auto pool = cuda::managed_default_memory_pool();

     // Allocate managed memory
     void* ptr = pool.allocate(stream, 1024);

     // Accessible from both host and device
     // Use memory...

     pool.deallocate(stream, ptr, 1024);
   }

``cuda::pinned_default_memory_pool``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _cccl-runtime-memory-pools-pinned-default:

``cuda::pinned_default_memory_pool()`` returns a non-owning reference to the default pinned (page-locked) host memory pool.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_default_pinned_pool(cuda::stream_ref stream) {
     // Get the default pinned memory pool
     auto pool = cuda::pinned_default_memory_pool();

     // Allocate pinned memory
     void* ptr = pool.allocate(stream, 1024);

     // Use for fast host-device transfers...

     pool.deallocate(stream, ptr, 1024);
   }

Notes on Default Pools
~~~~~~~~~~~~~~~~~~~~~~

- Default pools are created automatically by CUDA and shared across the application
- The pools are returned as non-owning references (``*_pool_ref`` types)
- Default pools use CUDA's default configuration and cannot be destroyed
- Multiple calls to the same getter function return references to the same pool
- Default pools are thread-safe and can be used concurrently from multiple threads
- Underlying CUDA default memory pools have 0 release threshold by default. First access to a default pool through one of the getters above will set the release threshold to the maximum value, unless previously modified by the user.

Memory Pool Properties
----------------------
.. _cccl-runtime-memory-pools-pool-properties:

``cuda::memory_pool_properties`` controls memory pool creation options:

- ``initial_pool_size`` - Initial size of the pool (default: 0)
- ``release_threshold`` - Threshold at which unused memory is released (default: no limit on the reserved memory)
- ``allocation_handle_type`` - Handle type for inter-process sharing (default: none)
- ``max_pool_size`` - Maximum size of the pool (default: no limit on the pool size)

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/devices>

   void create_pool_with_properties() {
     cuda::memory_pool_properties props{};
     props.initial_pool_size = 1024 * 1024;  // 1 MB initial size
     props.release_threshold = 10 * 1024 * 1024;  // Release if over 10 MB

     cuda::device_memory_pool pool{cuda::devices[0], props};
   }

Memory Pool Attributes
----------------------
.. _cccl-runtime-memory-pools-pool-attributes:

``cuda::memory_pool_attributes`` provides access to pool attributes for querying and configuration:

- ``release_threshold`` - Get/set the release threshold, which controls how much memory the pool can keep reserved, both used and unused
- ``reuse_follow_event_dependencies`` - Enable/disable reuse across streams with event dependencies
- ``reuse_allow_opportunistic`` - Enable/disable opportunistic reuse
- ``reuse_allow_internal_dependencies`` - Enable/disable reuse with internal dependencies
- ``reserved_mem_current`` - Query current reserved memory (read-only)
- ``used_mem_current`` - Query current used memory (read-only)
- ``reserved_mem_high`` - Get/set high watermark for reserved memory
- ``used_mem_high`` - Get/set high watermark for used memory

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/devices>

   void configure_pool_attributes() {
     cuda::device_memory_pool pool{cuda::devices[0]};

     // Set release threshold
     pool.set_attribute(cuda::memory_pool_attributes::release_threshold, 5 * 1024 * 1024);

     // Enable opportunistic reuse
     pool.set_attribute(cuda::memory_pool_attributes::reuse_allow_opportunistic, true);

     // Query current usage
     auto reserved = pool.attribute(cuda::memory_pool_attributes::reserved_mem_current);
     auto used = pool.attribute(cuda::memory_pool_attributes::used_mem_current);
   }

Pool Management
---------------
.. _cccl-runtime-memory-pools-pool-management:

Memory pools provide additional management functions:

- ``trim_to(min_bytes)`` - Release memory down to a minimum size
- ``enable/disable_access_from(devices)`` - Enable or disable access from specific devices (for peer access or access to host pinned memory)
- ``get()`` - Get the underlying ``cudaMemPool_t`` handle
- ``release()`` - Release ownership of the pool handle

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/devices>

   void manage_pool() {
     cuda::pinned_memory_pool pool{};

     // Enable access from all devices
     pool.enable_access_from(cuda::devices);

     // Trim pool to 1 MB minimum
     pool.trim_to(1024 * 1024);

     // Get native handle
     cudaMemPool_t handle = pool.get();
   }

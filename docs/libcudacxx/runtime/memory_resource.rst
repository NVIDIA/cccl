.. _cccl-runtime-memory-resource:

Memory Resources
================

The memory resource API provides property-checked, stream-ordered memory allocation interfaces for CUDA C++ developers. These APIs work seamlessly with :ref:`cuda::stream_ref <cccl-runtime-stream-stream-ref>` to enable stream-ordered memory management.

For detailed information about memory resource concepts and properties, see the :ref:`Extended API documentation <libcudacxx-extended-api-memory-resources>`.

``cuda::mr::any_resource``
---------------------------
.. _cccl-runtime-memory-resource-any-resource:

``cuda::mr::any_resource`` is a type-erased wrapper that owns any memory resource satisfying the specified properties. It's especially suited for use in container types that need to ensure the memory resource's lifetime exceeds the lifetime of the container.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/devices>
   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_any_resource(cuda::stream_ref stream) {
     // Wrap a device memory resource
    cuda::mr::any_resource<cuda::mr::device_accessible> resource{
      cuda::device_default_memory_pool(cuda::devices[0])
    };

     // Allocate memory
     void* ptr = resource.allocate(stream, 1024, 16);

     // Use memory...

     // Deallocate
     resource.deallocate(stream, ptr, 1024, 16);
   }

``cuda::mr::any_synchronous_resource``
---------------------------------------
.. _cccl-runtime-memory-resource-any-synchronous-resource:

``cuda::mr::any_synchronous_resource`` is a type-erased wrapper for synchronous memory resources (those that only support synchronous allocation/deallocation). It can be constructed from an ``any_resource``.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>

   void use_sync_resource() {
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> resource{
      cuda::mr::legacy_pinned_memory_resource{}
    };

     // Synchronous allocation
     void* ptr = resource.allocate_sync(1024, 16);

     // Use memory...

     // Synchronous deallocation
     resource.deallocate_sync(ptr, 1024, 16);
   }

``cuda::mr::resource_ref``
--------------------------
.. _cccl-runtime-memory-resource-resource-ref:

``cuda::mr::resource_ref`` is a non-owning, type-erased wrapper around a memory resource that supports stream-ordered allocation. It enables interfaces to specify properties of resources they expect without taking ownership of the resource.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/devices>
   #include <cuda/memory_resource>
   #include <cuda/stream>

   // indicates that the resource must be device-accessible
   void use_resource_ref(
     cuda::stream_ref stream,
     cuda::mr::resource_ref<cuda::mr::device_accessible> resource
   ) {
     // Allocate using the referenced resource
     void* ptr = resource.allocate(stream, 1024, 16);

     // Use memory...

     // Deallocate
     resource.deallocate(stream, ptr, 1024, 16);
   }

``cuda::mr::synchronous_resource_ref``
---------------------------------------
.. _cccl-runtime-memory-resource-synchronous-resource-ref:

``cuda::mr::synchronous_resource_ref`` is a non-owning, type-erased wrapper around a synchronous memory resource. It can be constructed from a ``resource_ref``.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>

   // indicates that the resource must be host-accessible
   void use_sync_ref(
     cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> resource
   ) {
     // Synchronous allocation
     void* ptr = resource.allocate_sync(1024, 16);

     // Use memory...

     // Synchronous deallocation
     resource.deallocate_sync(ptr, 1024, 16);
   }

``cuda::mr::shared_resource``
-----------------------------
.. _cccl-runtime-memory-resource-shared-resource:

``cuda::mr::shared_resource`` holds a reference-counted instance of a memory resource, allowing resources to be passed around with reference semantics while avoiding lifetime issues. This is particularly useful when a resource needs to be shared between multiple objects, like buffers allocated from the same resource.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_shared_resource(cuda::stream_ref stream) {
     // Create a shared resource
    auto shared_mr = cuda::mr::shared_resource{
      cuda::std::in_place_type<cuda::device_memory_pool>,
      cuda::devices[0]
    };

     // Copy the shared resource (shares ownership)
     auto shared_mr2 = shared_mr;

     // Both can be used independently
     void* ptr1 = shared_mr.allocate(stream, 1024, 16);
     void* ptr2 = shared_mr2.allocate(stream, 2048, 16);

     shared_mr.deallocate(stream, ptr1, 1024, 16);
     shared_mr2.deallocate(stream, ptr2, 2048, 16);
     // Resources are automatically cleaned up when last reference is destroyed
   }

``cuda::mr::synchronous_resource_adapter``
------------------------------------------
.. _cccl-runtime-memory-resource-synchronous-adapter:

``cuda::mr::synchronous_resource_adapter`` adapts a synchronous memory resource to work as a stream-ordered resource. If the underlying resource already supports stream-ordered allocation, it passes through the calls. Otherwise, it uses synchronous allocation/deallocation with proper stream synchronization.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>

   void adapt_sync_resource(cuda::stream_ref stream) {
     // Create a synchronous resource
    auto sync_mr = cuda::mr::legacy_pinned_memory_resource{};

     // Adapt it to work with streams
     auto adapted = cuda::mr::synchronous_resource_adapter{
       sync_mr
     };

     // Now can use with stream (will synchronize internally)
     void* ptr = adapted.allocate(stream, 1024, 16);

     // Use memory...

     // Deallocate (will synchronize stream before deallocation)
     adapted.deallocate(stream, ptr, 1024, 16);
   }

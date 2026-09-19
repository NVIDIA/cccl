.. _libcudacxx-extended-api-memory-resources-utilities:

Resource utilities
------------------

The ``cuda::mr`` memory resource system includes utilities that help manage resource lifetime and adapt synchronous
resources for stream-ordered usage. These utilities complement the type-erased wrappers in :ref:`resource wrappers <libcudacxx-extended-api-memory-resources-resource-ref>`.

shared_resource
~~~~~~~~~~~~~~~
.. _libcudacxx-extended-api-memory-resources-shared-resource:
.. _cccl-runtime-memory-resource-shared-resource:

``cuda::mr::shared_resource`` holds a reference-counted instance of a memory resource, allowing resources to be passed
around with shared ownership semantics while avoiding lifetime issues. This is useful when multiple objects, like
:ref:`cuda::buffer <cccl-runtime-buffer-buffer>`, must share the same resource instance.

``cuda::mr::make_shared_resource<Resource>(args...)`` constructs a ``shared_resource<Resource>`` that owns a newly
constructed ``Resource``. ``Resource`` must satisfy ``cuda::mr::synchronous_resource``. The resulting
``shared_resource`` supports stream-ordered allocation when ``Resource`` also satisfies ``cuda::mr::resource``.

.. code:: cpp

   #include <cuda/devices>
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

get_memory_resource
~~~~~~~~~~~~~~~~~~~
.. _libcudacxx-extended-api-memory-resources-get-memory-resource:

``cuda::mr::get_memory_resource`` is a customization point object that retrieves a memory resource from an object or
execution environment.

``cuda::mr::memory_resource_base<Derived>`` is a CRTP helper that makes a resource queryable by
``cuda::mr::get_memory_resource`` when the resource is used as part of a composed execution environment.

It supports three cases:

- If the object is itself a synchronous resource, it returns that object.
- If the object has a ``get_memory_resource()`` member, it calls that member.
- Otherwise, if an execution environment supports ``query(cuda::mr::get_memory_resource)``, it calls that query.

.. code:: cpp

   #include <cuda/memory_resource>
   #include <cuda/stream>
   #include <cuda/std/type_traits>

   template <class Env>
   void* allocate_from_env(Env& env, cuda::stream_ref stream, std::size_t size, std::size_t align) {
     auto&& resource = cuda::mr::get_memory_resource(env);
     using resource_type = cuda::std::remove_reference_t<decltype(resource)>;
     if constexpr (cuda::mr::resource<resource_type>) {
       return resource.allocate(stream, size, align);
     } else {
       return resource.allocate_sync(size, align);
     }
   }

synchronous_resource_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _libcudacxx-extended-api-memory-resources-synchronous-adapter:
.. _cccl-runtime-memory-resource-synchronous-adapter:

``cuda::mr::synchronous_resource_adapter`` adapts a synchronous memory resource to work as a stream-ordered resource.
If the underlying resource already supports stream-ordered allocation, it passes through the calls. Otherwise, it uses
synchronous allocation/deallocation with proper stream synchronization.

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

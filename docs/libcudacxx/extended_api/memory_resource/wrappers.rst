.. _libcudacxx-extended-api-memory-resources-wrappers:
.. _libcudacxx-extended-api-memory-resources-resource-ref:
.. _cccl-runtime-memory-resource-resource-ref:
.. _libcudacxx-memory-resource-any-resource:
.. _libcudacxx-memory-resource-any-async-resource:

Type-erased resource wrappers
-----------------------------

With the property design depicted in :ref:`cuda::get_property <libcudacxx-extended-api-memory-resources-properties>`,
a library has flexibility in checking constraints and querying custom properties. However, there is also a cost in
providing function templates for a potentially wide range of inputs. Depending on the number of different memory
resources, both compile time and binary size might increase considerably.

The type-erased wrappers let you coalesce such APIs into a single function. Both ``resource_ref`` and ``any_resource``
preserve property constraints while erasing the concrete resource type.

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Wrapper
     - Ownership
     - Typical use
   * - ``cuda::mr::resource_ref``
     - Non-owning
     - Accept references/pointers without extending lifetime
   * - ``cuda::mr::any_resource``
     - Owning
     - Store a resource with the object that uses it

Common usage
~~~~~~~~~~~~

Both wrappers provide a non-templated API surface for allocation and property queries. Choose which wrapper to use based
on ownership, then the usage patterns are the same.

``resource_ref`` is constructible from any non-const reference or pointer to a memory resource that satisfies
``cuda::mr::{synchronous_}resource``. ``any_resource`` is constructible from a resource object and takes ownership of it.

Properties may be passed to both wrappers just as with ``cuda::mr::resource_with``.

.. code:: cpp

   void* do_allocate(cuda::mr::resource_ref<> resource, cuda::stream_ref stream, std::size_t size, std::size_t align) {
       return resource.allocate(stream, size, align);
   }

   void* do_allocate_owned(cuda::mr::any_resource<> resource, cuda::stream_ref stream, std::size_t size, std::size_t align) {
       return resource.allocate(stream, size, align);
   }

   my_memory_resource resource;
   my_memory_resource* pointer_to_resource = &resource;

   void* from_reference = do_allocate(resource, stream, 1337, 256);
   void* from_ptr = do_allocate(pointer_to_resource, stream, 1337, 256);
   void* from_owned = do_allocate_owned(cuda::mr::any_resource<>{resource}, stream, 1337, 256);

resource_ref
~~~~~~~~~~~~
.. _libcudacxx-extended-api-memory-resources-resource-ref-wrapper:

``cuda::mr::resource_ref`` is the non-owning, type-erased wrapper. Prefer it when the caller controls the resource
lifetime.

.. code:: cpp

   struct required_alignment{};
   void* do_allocate_with_alignment(cuda::mr::resource_ref<required_alignment> resource, cuda::stream_ref stream, std::size_t size) {
       return resource.allocate(stream, size, cuda::mr::get_property(resource, required_alignment));
   }

However, the type erasure comes with the cost that arbitrary properties cannot be queried from either wrapper:

.. code:: cpp

   struct required_alignment{};
   void* buggy_allocate_with_alignment(cuda::mr::resource_ref<> resource, cuda::stream_ref stream, std::size_t size) {
       if constexpr (cuda::has_property<required_alignment>) { // BUG: This will always be false
           return resource.allocate(stream, size, cuda::mr::get_property(resource, required_alignment));
       } else {
           return resource.allocate(stream, size, my_default_alignment);
       }
   }

So, choose wisely. If your library has a well-defined set of fixed properties that you expect to always be available,
then ``cuda::mr::{synchronous_}resource_ref`` is an amazing tool to improve compile times and binary size. If you need a
flexible interface then constraining a template argument through ``cuda::mr::{synchronous_}resource_with`` is the proper solution.

any_resource
~~~~~~~~~~~~
.. _libcudacxx-extended-api-memory-resources-any-resource-wrapper:
.. _libcudacxx-extended-api-memory-resources-any-resource:
.. _cccl-runtime-memory-resource-any-resource:

``cuda::mr::any_resource`` is the owning counterpart. It is especially suited for containers that must ensure the
resource outlives the container.

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

Synchronous variants
~~~~~~~~~~~~~~~~~~~~

The synchronous wrappers mirror the same ownership split: ``synchronous_resource_ref`` is non-owning, while
``any_synchronous_resource`` owns the resource instance.

``cuda::mr::any_synchronous_resource``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _libcudacxx-extended-api-memory-resources-any-synchronous-resource:
.. _cccl-runtime-memory-resource-any-synchronous-resource:

``cuda::mr::any_synchronous_resource`` is a type-erased wrapper for synchronous memory resources (those that only support
synchronous allocation/deallocation). It can be constructed from an ``any_resource``.

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

``cuda::mr::synchronous_resource_ref``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _cccl-runtime-memory-resource-synchronous-resource-ref:

``cuda::mr::synchronous_resource_ref`` provides the same type-erased reference behavior as ``resource_ref``, but targets
resources that only offer synchronous allocation and deallocation.

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

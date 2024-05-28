.. _libcudacxx-extended-api-memory-resources-resource-ref:

``cuda::resource_ref``: a type-constrained resource wrapper
------------------------------------------------------------

With the property design depicted in :ref:`cuda::get_property <libcudacxx-extended-api-memory-resources-properties>`,
a library has flexibility in checking constraints and querying custom properties. However, there is also a cost in
providing function templates for a potentially wide range of inputs. Depending on the number of different memory
resources, both compile time and binary size might increase considerably.

The type-erased ``resource_ref`` and ``async_resource_ref`` resource wrappers aid in efficiently coalescing such APIs
into a single function.

.. code:: cpp

   void* do_allocate_async(cuda::mr::async_resource_ref<> resource, std::size_t size, std::size_t align, cuda::stream_ref stream) {
       return resource.allocate_async(size, align, stream);
   }

   my_async_memory_resource resource;
   my_async_memory_resource* pointer_to_resource = &resource;

   void* from_reference = do_allocate_async(resource, 1337, 256, cuda::stream_ref{});
   void* from_ptr = do_allocate_async(pointer_to_resource, 1337, 256, cuda::stream_ref{});

Note that ``do_allocate_async`` is not a template anymore but a plain old function. The wrapper
``cuda::mr::{async_}resource_ref<>`` is constructible from any non-const reference or pointer to a memory resource that
satisfies ``cuda::mr::{async_}resource``.

Properties may also be passed to ``cuda::mr::{async_}resource_ref`` just as with ``cuda::mr::resource_with``.

.. code:: cpp

   struct required_alignment{};
   void* do_allocate_async_with_alignment(cuda::mr::async_resource_ref<required_alignment> resource, std::size_t size, cuda::stream_ref stream) {
       return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
   }

However, the type erasure comes with the cost that arbitrary properties cannot be queried:

.. code:: cpp

   struct required_alignment{};
   void* buggy_allocate_async_with_alignment(cuda::mr::async_resource_ref<> resource, std::size_t size, cuda::stream_ref stream) {
       if constexpr (cuda::has_property<required_alignment>) { // BUG: This will always be false
           return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
       } else {
           return resource.allocate_async(size, my_default_alignment, stream);
       }
   }

So, choose wisely. If your library has a well-defined set of fixed properties that you expect to always be available,
then ``cuda::mr::{async_}resource_ref`` is an amazing tool to improve compile times and binary size. If you need a
flexible interface then constraining through ``cuda::mr::{async_}resource_with`` is the proper solution.

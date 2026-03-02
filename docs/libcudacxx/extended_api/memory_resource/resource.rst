.. _libcudacxx-extended-api-memory-resources-resource:
.. _libcudacxx-extended-api-memory-resources-synchronous-resource:

The ``cuda::synchronous_resource`` concept
-------------------------------------------

The `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__ feature provides only a
single ``allocate`` interface, which is sufficient for homogeneous memory systems. However, CUDA provides both
synchronous and `stream-ordered allocation <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator>`__.

With `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__ there is no way to tell
whether a memory resource can utilize stream-ordered allocations. Even if the application knows it can, there is no way
to properly tell the memory resource to use stream-ordered allocation. Ideally, this should not be something discovered
through an assert at run time, but should be checked by the compiler.

Because asynchronous memory management is critical for performance, ``cuda::mr::resource`` defaults to stream-ordered interface provided by ``allocate`` / ``deallocate``.
For cases where stream-ordered allocation is not possible, ``cuda::mr::synchronous_resource`` is provided.

The ``cuda::mr::synchronous_resource`` concept provides basic type checks to ensure that a given memory resource provides the
expected ``allocate_sync`` / ``deallocate_sync`` interface and is also equality comparable, which covers the whole API surface of
`std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__.
See below for different memory resources and potential pitfalls.

To demonstrate, the following example defines several resources, only some of which are valid implementations of the
``cuda::mr::synchronous_resource`` concept. The ``static_assertions`` will result in compile-time errors for the invalid resources.

.. code:: cpp

   struct valid_resource {
     void* allocate_sync(std::size_t, std::size_t) { return nullptr; }
     void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
     bool operator==(const valid_resource&) const { return true; }
     // NOTE: C++20 thankfully added operator rewrite rules so defining operator!= is not required.
     // However, if compiled with C++14 / C++17, operator != must also be defined.
     bool operator!=(const valid_resource&) const { return false; }
   };
   static_assert(cuda::mr::resource<valid_resource>, "");

   struct invalid_argument {};
   struct invalid_allocate_argument {
     void* allocate_sync(invalid_argument, std::size_t) { return nullptr; }
     void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
     bool operator==(const invalid_allocate_argument&) { return true; }
   };
   static_assert(!cuda::mr::resource<invalid_allocate_argument>, "");

   struct invalid_allocate_return {
     int allocate_sync(std::size_t, std::size_t) { return 42; }
     void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
     bool operator==(const invalid_allocate_return&) { return true; }
   };
   static_assert(!cuda::mr::resource<invalid_allocate_return>, "");

   struct invalid_deallocate_argument {
     void* allocate_sync(std::size_t, std::size_t) { return nullptr; }
     void deallocate_sync(void*, invalid_argument, std::size_t) noexcept {}
     bool operator==(const invalid_deallocate_argument&) { return true; }
   };
   static_assert(!cuda::mr::resource<invalid_deallocate_argument>, "");

   struct non_comparable {
     void* allocate_sync(std::size_t, std::size_t) { return nullptr; }
     void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
   };
   static_assert(!cuda::mr::resource<non_comparable>, "");

   struct non_eq_comparable {
     void* allocate_sync(std::size_t, std::size_t) { return nullptr; }
     void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
     bool operator!=(const non_eq_comparable&) { return false; }
   };
   static_assert(!cuda::mr::synchronous_resource<non_eq_comparable>, "");

In addition to the `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`_ interface the
``cuda::mr::resource`` concept verifies that a memory resource also satisfies the ``allocate`` /
``deallocate`` interface. Requiring both the PMR interface and the async interface is a deliberate design decision.

.. code:: cpp

   struct valid_resource {
     void* allocate_sync(std::size_t, std::size_t) { return nullptr; }
     void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
     void* allocate(cuda::stream_ref, std::size_t, std::size_t) { return nullptr; }
     void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
     bool operator==(const valid_resource&) const { return true; }
     bool operator!=(const valid_resource&) const { return false; }
   };
   static_assert(cuda::mr::resource<valid_resource>, "");

A library can easily decide whether to use the async interface:

.. code:: cpp

   template<class MemoryResource>
       requires cuda::mr::synchronous_resource<MemoryResource>
   void* allocate_maybe_sync(cuda::stream_ref stream, MemoryResource& resource, std::size_t size, std::size_t align) {
       if constexpr(cuda::mr::resource<MemoryResource>) {
           return resource.allocate(stream, size, align);
       } else {
           return resource.allocate_sync(size, align);
       }
   }

.. rubric:: Putting them together

Applications and libraries may want to combine type checks for arbitrary properties with the ``{synchronous_}resource``
concept. The ``{synchronous_}resource_with`` concept allows checking resources for arbitrary properties.

.. code:: cpp

   struct required_alignment{
       using value_type = std::size_t;
   };
   struct my_memory_resource {
       void* allocate_sync(std::size_t, std::size_t) { return nullptr; }
       void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
       bool operator==(const my_memory_resource&) const { return true; }
       bool operator!=(const my_memory_resource&) const { return false; }

       friend constexpr std::size_t get_property(const my_memory_resource& resource, required_alignment) noexcept { return resource.required_alignment; }

       std::size_t required_alignment;
   };

   template<class MemoryResource>
       requires cuda::mr::resource<MemoryResource>
   void* allocate_maybe_sync_check_alignment(MemoryResource& resource, cuda::stream_ref stream, std::size_t size) {
       if constexpr(cuda::mr::resource_with<MemoryResource, required_alignment>) {
           return resource.allocate(stream, size, get_property(resource, required_alignment));
       } else if constexpr (cuda::mr::resource<MemoryResource>) {
           return resource.allocate(stream, size, my_default_alignment);
       } else if constexpr (cuda::mr::synchronous_resource_with<MemoryResource, required_alignment>) {
           return resource.allocate_sync(size, get_property(resource, required_alignment));
       } else {
           return resource.allocate_sync(size, my_default_alignment);
       }
   }

   // Potentially more concise
   template<class MemoryResource>
       requires cuda::mr::resource<MemoryResource>
   void* allocate_maybe_sync_check_alignment2(MemoryResource& resource, cuda::stream_ref stream, std::size_t size) {
       constexpr std::size_t align = cuda::mr::resource_with<MemoryResource, required_alignment>
                                   ? get_property(resource, required_alignment)
                                   : my_default_alignment;
       if constexpr(cuda::mr::resource<MemoryResource>) {
           return resource.allocate(stream, size, align);
       } else {
           return resource.allocate_sync(size, align);
       }
   }

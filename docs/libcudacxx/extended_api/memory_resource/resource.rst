.. _libcudacxx-extended-api-memory-resources-resource:

The ``cuda::resource`` concept
-------------------------------

The `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__ feature provides only a
single ``allocate`` interface, which is sufficient for homogeneous memory systems. However, CUDA provides both
synchronous and `stream-ordered allocation <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator>`__.

With `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__ there is no way to tell
whether a memory resource can utilize stream-ordered allocations. Even if the application knows it can, there is no way
to properly tell the memory resource to use stream-ordered allocation. Ideally, this should not be something discovered
through an assert at run time, but should be checked by the compiler.

The ``cuda::mr::resource`` concept provides basic type checks to ensure that a given memory resource provides the
expected ``allocate`` / ``deallocate`` interface and is also equality comparable, which covers the whole API surface of
`std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`__.
See below for different memory resources and potential pitfals.

To demonstrate, the following example defines several resources, only some of which are valid implementations of the
``cuda::mr::resource`` concept. The ``static_assertion``'s will result in compile-time errors for the invalid resources.

.. code:: cpp

   struct valid_resource {
     void* allocate(std::size_t, std::size_t) { return nullptr; }
     void deallocate(void*, std::size_t, std::size_t) noexcept {}
     bool operator==(const valid_resource&) const { return true; }
     // NOTE: C++20 thankfully added operator rewrite rules so defining operator!= is not required.
     // However, if compiled with C++14 / C++17, operator != must also be defined.
     bool operator!=(const valid_resource&) const { return false; }
   };
   static_assert(cuda::mr::resource<valid_resource>, "");

   struct invalid_argument {};
   struct invalid_allocate_argument {
     void* allocate(invalid_argument, std::size_t) { return nullptr; }
     void deallocate(void*, std::size_t, std::size_t) noexcept {}
     bool operator==(const invalid_allocate_argument&) { return true; }
   };
   static_assert(!cuda::mr::resource<invalid_allocate_argument>, "");

   struct invalid_allocate_return {
     int allocate(std::size_t, std::size_t) { return 42; }
     void deallocate(void*, std::size_t, std::size_t) noexcept {}
     bool operator==(const invalid_allocate_return&) { return true; }
   };
   static_assert(!cuda::mr::resource<invalid_allocate_return>, "");

   struct invalid_deallocate_argument {
     void* allocate(std::size_t, std::size_t) { return nullptr; }
     void deallocate(void*, invalid_argument, std::size_t) noexcept {}
     bool operator==(const invalid_deallocate_argument&) { return true; }
   };
   static_assert(!cuda::mr::resource<invalid_deallocate_argument>, "");

   struct non_comparable {
     void* allocate(std::size_t, std::size_t) { return nullptr; }
     void deallocate(void*, std::size_t, std::size_t) noexcept {}
   };
   static_assert(!cuda::mr::resource<non_comparable>, "");

   struct non_eq_comparable {
     void* allocate(std::size_t, std::size_t) { return nullptr; }
     void deallocate(void*, std::size_t, std::size_t) noexcept {}
     bool operator!=(const non_eq_comparable&) { return false; }
   };
   static_assert(!cuda::mr::resource<non_eq_comparable>, "");

In addition to the `std::pmr::memory_resource <https://en.cppreference.com/w/cpp/header/memory_resource>`_ interface the
``cuda::mr::async_resource`` concept verifies that a memory resource also satisfies the ``allocate_async`` /
``deallocate_async`` interface. Requiring both the PMR interface and the async interface is a deliberate design decision.

.. code:: cpp

   struct valid_resource {
     void* allocate(std::size_t, std::size_t) { return nullptr; }
     void deallocate(void*, std::size_t, std::size_t) noexcept {}
     void* allocate_async(std::size_t, std::size_t, cuda::stream_ref) { return nullptr; }
     void deallocate_async(void*, std::size_t, std::size_t, cuda::stream_ref) {}
     bool operator==(const valid_resource&) const { return true; }
     bool operator!=(const valid_resource&) const { return false; }
   };
   static_assert(cuda::mr::async_resource<valid_resource>, "");

A library can easily decide whether to use the async interface:

.. code:: cpp

   template<class MemoryResource>
       requires cuda::mr::resource<MemoryResource>
   void* maybe_allocate_async(MemoryResource& resource, std::size_t size, std::size_t align, cuda::stream_ref stream) {
       if constexpr(cuda::mr::async_resource<MemoryResource>) {
           return resource.allocate_async(size, align, stream);
       } else {
           return resource.allocate(size, align);
       }
   }

.. rubric:: Putting them together

Applications and libraries may want to combine type checks for arbitrary properties with the ``{async_}resource``
concept. The ``{async_}resource_with`` concept allows checking resources for arbitrary properties.

.. code:: cpp

   struct required_alignment{
       using value_type = std::size_t;
   };
   struct my_memory_resource {
       void* allocate(std::size_t, std::size_t) { return nullptr; }
       void deallocate(void*, std::size_t, std::size_t) noexcept {}
       bool operator==(const my_memory_resource&) const { return true; }
       bool operator!=(const my_memory_resource&) const { return false; }

       friend constexpr std::size_t get_property(const my_memory_resource& resource, required_alignment) noexcept { return resource.required_alignment; }

       std::size_t required_alignment;
   };

   template<class MemoryResource>
       requires cuda::mr::resource<MemoryResource>
   void* maybe_allocate_async_check_alignment(MemoryResource& resource, std::size_t size, cuda::stream_ref stream) {
       if constexpr(cuda::mr::async_resource_with<MemoryResource, required_alignment>) {
           return resource.allocate_async(size, get_property(resource, required_alignment), stream);
       } else if constexpr (cuda::mr::async_resource<MemoryResource>) {
           return resource.allocate_async(size, my_default_alignment, stream);
       } else if constexpr (cuda::mr::resource_with<MemoryResource, required_alignment>) {
           return resource.allocate(size, get_property(resource, required_alignment));
       } else {
           return resource.allocate(size, my_default_alignment);
       }
   }

   // Potentially more concise
   template<class MemoryResource>
       requires cuda::mr::resource<MemoryResource>
   void* maybe_allocate_async_check_alignment2(MemoryResource& resource, std::size_t size, cuda::stream_ref stream) {
       constexpr std::size_t align = cuda::mr::resource_with<MemoryResource, required_alignment>
                                   ? get_property(resource, required_alignment)
                                   : my_default_alignment;
       if constexpr(cuda::mr::async_resource<MemoryResource>) {
           return resource.allocate_async(size, align, stream);
       } else {
           return resource.allocate(size, align);
       }
   }

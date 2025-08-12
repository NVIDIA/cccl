.. _libcudacxx-extended-api-memory-resources-properties:

Properties
----------

Modern C++ programs use the type system to verify statically known properties of the code. It is undesirable to use
run-time assertions to verify whether a function that accesses device memory is passed a memory resource that provides
device-accessible allocations. A property system is provided for this purpose.

To tell the compiler that memory provided by ``my_memory_resource`` is device accessible, use the
``cuda::mr::device_accessible`` tag type with a free function ``get_property`` to declare that the resource provides
device-accessible memory.

.. code:: cpp

   struct my_memory_resource {
       friend constexpr void get_property(const my_memory_resource&, cuda::mr::device_accessible) noexcept {}
   };

A library can constrain interfaces with ``cuda::has_property`` to require that a passed memory resource provides the
right kind of memory

`See it on Godbolt <https://godbolt.org/z/5hjoEnerb>`__

.. code:: cpp

   template<class MemoryResource>
       requires cuda::has_property<MemoryResource, cuda::mr::device_accessible>
   void function_that_dispatches_to_device(MemoryResource& resource);

If C++20 is not available, the function can instead be constrained via SFINAE

`See it on Godbolt  <https://godbolt.org/z/11sGbr333>`__

.. code:: cpp

   template<class MemoryResource, class = cuda::std::enable_if_t<cuda::has_property<MemoryResource, cuda::mr::device_accessible>>>
   void function_that_dispatches_to_device(MemoryResource& resource);

For now, libcu++ provides various commonly used properties:

-  ``cuda::mr::device_accessible`` and ``cuda::mr::host_accessible`` indicate whether memory allocated using a
   memory resource is accessible from host or device respectively.

More properties may be added as the library and the hardware capabilities evolve. However, a user library is free to
define custom properties.

Note that currently the libcu++ provided properties are stateless. However, properties can also provide stateful
information that is retrieved via the ``get_property`` free function. In order to communicate the desired type of the
carried state, a stateful property must define the ``value_type`` alias. A library can constrain interfaces that
require a stateful property with ``cuda::has_property_with`` as shown in the example below

`See it on Godbolt  <https://godbolt.org/z/11sGbr333>`__

.. code:: cpp

   struct required_alignment{
       using value_type = std::size_t;
   };
   struct my_memory_resource {
       friend constexpr std::size_t get_property(const my_memory_resource& resource, required_alignment) noexcept { return resource.required_alignment; }

       std::size_t required_alignment;
   };
   static_assert(cuda::has_property_with<my_memory_resource, required_alignment, std::size_t>);

   template<class MemoryResource>
   void* allocate_check_alignment(MemoryResource& resource, std::size_t size) {
       if constexpr(cuda::has_property_with<MemoryResource, required_alignment, std::size_t>) {
           return resource.allocate_sync(size, get_property(resource, required_alignment{}));
       } else {
           // Use default alignment
           return resource.allocate_sync(size, 42);
       }
   }

In generic code it is often desirable to propagate properties from a base type to a derived type, without knowing the
base type at all. This common use case is covered by ``cuda::forward_property``, a simple CRTP template.

.. code:: cpp

   template<class MemoryResource>
   class logging_resource : cuda::forward_property<logging_resource<MemoryResource>, MemoryResource> {
       MemoryResource base;
   public:
       void* allocate_sync(std::size_t size, std::size_t alignment) {
           std::cout << "allocating\n";
           return base.allocate_sync(size, alignment);
       }
       void deallocate_sync(void* ptr, std::size_t size, std::size_t alignment) noexcept {
           std::cout << "deallocating\n";
           return base.deallocate(ptr, size, alignment);
       }
   }

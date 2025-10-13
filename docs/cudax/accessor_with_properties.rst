.. _libcudacxx-extended-api-accessor-with-properties:

``mdspan`` Accessor with Properties
===================================

**Accessor with Properties** is a CUDA-specific ``mdspan`` accessor that allows users to exploit advanced memory access capabilities by replacing the ``mdspan`` default behavior.

The accessor allows to specify the following properties:

- **Pointer aliasing policy**, see `Pointer Aliasing Policy`_ for more details
- **Memory alignment**, see `Memory Alignment`_ for more details
- **Cache eviction policy**, see `Cache Eviction Policy`_ for more details
- **Prefetch size**, see `Prefetch Size`_ for more details

Types and Utilities
-------------------

The **main class** to describe the memory access properties

.. code:: cpp

  template <typename ElementType,
            typename AliasingPolicy,
            typename Alignment,
            typename Eviction,
            typename Prefetch>
  class accessor_with_properties;

A **proxy class** to allow distinguishing between load and store operations

.. code:: cpp

  template <typename ElementType,
            typename Restrict,
            typename Alignment,
            typename Eviction,
            typename Prefetch>
  class accessor_reference;

A **factory** to simplify the creation of an accessor with properties. It supports a variable number of properties and without enforcing a specific order

.. code:: cpp

  template <typename ElementType, typename... UserProperties>
  auto make_accessor_with_properties(UserProperties... properties) noexcept;

  template <typename ElementType, typename... UserProperties>
  auto add_properties(UserProperties... properties) noexcept;

A set of **predefined accessors**:

.. code:: cpp

  cuda::streaming_accessor
  cuda::cache_all_accessor

Pointer Aliasing Policy
-----------------------

The *aliasing policy* specifies how pointers are resolved by the compiler when they are used in an expression

- ``aliasing_policy::restrict``: **No aliasing**. Pointers doesn't overlap (**default**)
- ``aliasing_policy::may_alias``: **May alias**. Pointers may overlap

Memory Alignment
----------------

Specifies the alignment of the data, see
:ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-access-shapes-aligned-size>`. **Default**: ``alignof(ElementType)``

Cache Eviction Policy
---------------------

Cache eviction policies determine the order in which cache entries are removed when the cache reaches its capacity

- ``eviction_policy::first``: **Evict first**. Data will likely be evicted when cache eviction is required. This policy is suitable for streaming data
- ``eviction_policy::normal``: **Normal eviction policy**. It maps to a standard memory access (**default**)
- ``eviction_policy::last``:   **Evict last**. Data will likely be evicted only after other data with ``evict_normal`` or ``evict_first`` eviction priority is already evicted. This policy is suitable for persistent data
- ``eviction_policy::last_use``:      **Last use**. Data that is read can be invalidated even if dirty
- ``eviction_policy::no_allocation``: **No allocation**. Do not allocate data to cache. This policy is suitable for streaming data

Prefetch Size
-------------

The *prefetch size* is a hint to fetch additional data of the specified size into the L2 cache level

- ``prefetch_size::no_prefetch``: **No prefetch**
- ``prefetch_size::default``:   **Default prefetch** (**default**)
- ``prefetch_size::bytes_64``: **64 bytes prefetch**
- ``prefetch_size::bytes_128``: **128 bytes prefetch**
- ``prefetch_size::bytes_256``: **256 bytes prefetch**

Predefined Accessors
--------------------

+--------------------------------------+----------------------------------------+---------------------------------------+------------------------------------------+
|                                      |                                        | ``cub`` equivalent                                                               |
|                                      |                                        +---------------------------------------+------------------------------------------+
| **Name**                             | Properties                             | ``CacheLoadModifier``                 | ``CacheStoreModifier``                   |
+======================================+========================================+=======================================+==========================================+
| *>* ``streaming_accessor``           | ``eviction_policy::first``             | ``cub::LOAD_CS``                      | ``cub::STORE_CS``                        |
+--------------------------------------+----------------------------------------+---------------------------------------+------------------------------------------+
| *>* ``cache_all_accessor``           | ``memory_consistency_scope::gpu``      | ``cub::LOAD_CG``                      | ``cub::STORE_CG``                        |
+--------------------------------------+----------------------------------------+---------------------------------------+------------------------------------------+

**Example**:

[Example description]

.. code:: cpp

  #include <cuda/std/mdspan>
  #include <cuda/__mdspan/accessors_with_properties.h>

  int main() {
      auto x1 = cuda::make_accessor_with_properties<int>(cuda::eviction_policy::first);

      auto x2 = cuda::make_accessor_with_properties<int>(cuda::prefetch_size::no_prefetch,
                                                         cuda::eviction_policy::first);

      auto x3 = cuda::make_accessor_with_properties<const int>(cuda::aliasing_policy::restrict,
                                                               cuda::eviction_policy::last_use);

      auto x4 = cuda::make_accessor_with_properties<const int>(cuda::access_property::normal{},
                                                               cuda::eviction_policy::first);
      int* ptr;
      auto mapping = cuda::std::layout_right::mapping{cuda::std::extents{10}};
      auto mdspan1 = cuda::std::mdspan(ptr, mapping, cuda::streaming_accessor<int>{});
      auto mdspan2 = cuda::std::mdspan(ptr, mapping, cuda::cache_all_accessor<int>{});
      auto mdspan6 = cuda::std::mdspan(ptr, mapping, x4);

      // duplicate eviction policy
      // auto w = cuda::make_accessor_with_properties<int>(cuda::eviction_policy::first,
      //                                                   cuda::eviction_policy::normal);
  }

`See it on Godbolt TODO`

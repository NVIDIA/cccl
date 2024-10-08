.. _libcudacxx-extended-api-accessor-with-properties:

Accessor with Properties
========================

.. code:: cuda

  template <typename ElementType,
            typename AliasingPolicy,
            typename Alignment,
            typename Eviction,
            typename Scope,
            typename Prefetch,
            typename CacheHint>
  class accessor_with_properties;

.. code:: cuda

  template <typename ElementType, typename... UserProperties>
  auto make_accessor_with_properties(UserProperties... properties) noexcept;


Cache Eviction Policy
---------------------

Cache eviction policies determine the order in which cache entries are removed when the cache reaches its capacity

- ``eviction_policy::first``: **Evict first**. Data will likely be evicted when cache eviction is required. This policy is suitable for streaming data
- ``eviction_policy::normal``: **Default eviction policy**. It maps to a standard memory access
- ``eviction_policy::last``:   **Evict last**. Data will likely be evicted only after other data with ``evict_normal`` or ``evict_first`` eviction priotity is already evicted. This policy is suitable for persistent data
- ``eviction_policy::last_use``:      **Last use**. Data that is read can be invalidated even if dirty
- ``eviction_policy::no_allocation``: **No allocation**. Do not allocate data to cache. This policy is suitable for streaming data

Memory Consistency Scope
------------------------

The *memory consistency scope* defines the set of threads in which data is visible and consistent between reads and
writes

- ``memory_consistency_scope::none``:  the memory consistency scope is *not specified*  (*default*)
- ``memory_consistency_scope::cta``: the scope is limited to threads within the *same CTA/Thread Block*
- ``memory_consistency_scope::cluster``: the scope is limited to threads within the *same Thread Cluster*
- ``memory_consistency_scope::gpu``: the scope is limited to threads within the *same GPU*
- ``memory_consistency_scope::system``: the scope is *not limited*. It can interact with any thread in the system

Prefetch Size
-------------

The *prefetch size* is a hint to fetch additional data of the specified size into the L2 cache level

- ``prefetch_size::no_prefetch``: **No prefetch**  (*default*)
- ``prefetch_size::bytes_64``: **64 bytes prefetch**
- ``prefetch_size::bytes_128``: **128 bytes prefetch**
- ``prefetch_size::bytes_256``: **256 bytes prefetch**

Pointer Aliasing Policy
-----------------------

The *aliasing policy* specifies how pointers are resolved by the compiler when they are used in an expression

- ``aliasing_policy::restrict``: **No aliasing**. Pointers doesn't overlap (*default*)
- ``aliasing_policy::may_alias``: **May alias**. Pointers may overlap

Memory Alignment
----------------

Specifies the alignment of the data, see
:ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-access-shapes-aligned-size>`. Default: ``alignof(T)``

Cache Hint
----------

Specifies a hint to the L2 cache, see
:ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`. Default: ``cuda::access_property::global``

Predefined Accessors
--------------------

+---------------------------------+--------------------------------------+---------------------------------------+------------------------------------------+
| Name                            | Properties                           | ``cub::CacheLoadModifier`` equivalent | ``cub::CacheStoreModifier`` equivalent   |
+=================================+======================================+=======================================+==========================================+
| ``streaming_accessor``          | ``eviction_policy::first``           | ``cub::LOAD_CS``                      | ``cub::STORE_CS``                        |
+---------------------------------+--------------------------------------+---------------------------------------+------------------------------------------+
| ``cache_all_accessor``          | ``memory_consistency_scope::gpu``    | ``cub::LOAD_CG``                      | ``cub::STORE_CG``                        |
+---------------------------------+--------------------------------------+---------------------------------------+------------------------------------------+
| ``cache_global_accessor``       | ``memory_consistency_scope::cta``    | ``cub::LOAD_CA``                      | ``cub::STORE_WB``                        |
+---------------------------------+--------------------------------------+---------------------------------------+------------------------------------------+
| ``cache_invalidation_accessor`` | ``memory_consistency_scope::system`` | ``cub::LOAD_CV``                      | ``cub::STORE_WT``                        |
+---------------------------------+--------------------------------------+---------------------------------------+------------------------------------------+
| ``read_only_accessor``          | ``const T``                          | ``cub::LOAD_LDG``                     | N/A                                      |
+---------------------------------+--------------------------------------+---------------------------------------+------------------------------------------+

.. note::

   The function is only constexpr from C++14 onwards

**Example**:

[Example description]

.. code:: cuda

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
      auto mdspan3 = cuda::std::mdspan(ptr, mapping, cuda::cache_global_accessor<int>{});
      auto mdspan4 = cuda::std::mdspan(ptr, mapping, cuda::cache_invalidation_accessor<int>{});
      auto mdspan5 = cuda::std::mdspan(ptr, mapping, cuda::read_only_accessor<int>{});
      auto mdspan6 = cuda::std::mdspan(ptr, mapping, x4);

      // duplicate eviction policy
      // auto w = cuda::make_accessor_with_properties<int>(cuda::eviction_policy::first,
      //                                                   cuda::eviction_policy::normal);
  }

`See it on Godbolt TODO`

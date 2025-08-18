.. _libcudacxx-extended-api-memory-aligned-size:

``cuda::aligned_size_t``
========================

Defined in headers ``<cuda/memory>``, ``<cuda/barrier>`` and ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::std::size_t Alignment>
   struct cuda::aligned_size_t {
     static constexpr cuda::std::size_t align = Align;
     cuda::std::size_t value;
     __host__ __device__ explicit constexpr aligned_size(cuda::std::size_t size);
     __host__ __device__ constexpr operator cuda::std::size_t();
   };

The class template ``cuda::aligned_size_t`` is a *shape* representing an extent of bytes with a statically
defined (address and size) alignment.

*Preconditions*:

-  The *address* of the extent of bytes must be aligned to an ``Alignment`` alignment boundary.
-  The *size* of the extent of bytes must be a multiple of the ``Alignment``.

Template Parameters
-------------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Alignment``
     - The address and size alignment of the byte extent.

Data Members
------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``align``
     - The alignment of the byte extent.
   * - ``value``
     - The size of the byte extent.

Member Functions
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``(constructor)``
     - Constructs an *aligned size*. If the ``size`` is not a multiple of ``Alignment`` the behavior is undefined.
   * - ``(destructor)``
     - Trivial implicit destructor.
   * - ``operator=``
     - Trivial implicit copy/move.
   * - ``operator cuda::std::size_t``
     - Implicit conversion to `cuda::std::size_t <https://en.cppreference.com/w/cpp/types/size_t>`__.

Notes
-----

If ``Alignment`` is not a `valid alignment <https://en.cppreference.com/w/c/language/object#Alignment>`_,
the behavior is undefined.

Example
-------

.. code:: cuda

   #include <cuda/memory>

   __global__ void example_kernel(void* dst, void* src, size_t size) {
     cuda::barrier<cuda::thread_scope_system> bar;
     init(&bar, 1);

     // Implementation cannot make assumptions about alignment.
     cuda::memcpy_async(dst, src, size, bar);

     // Implementation can assume that dst and src are 16-bytes aligned,
     // and that size is a multiple of 16, and may optimize accordingly.
     cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(size), bar);

     bar.arrive_and_wait();
   }

`See it on Godbolt <https://godbolt.org/z/PWGdfTd7d>`_

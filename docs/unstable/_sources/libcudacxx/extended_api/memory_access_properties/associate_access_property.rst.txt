.. _libcudacxx-extended-api-memory-access-properties-associate-access-property:

``cuda::associate_access_property``
===================================

Defined in header ``<cuda/annotated_ptr>``.

.. code:: cuda

   template <typename T, typename Property>
   [[nodiscard]] __host__ __device__
   T* associate_access_property(T* ptr, Property prop) noexcept;

Associate an :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>` to the input pointer, such that subsequent memory operations with the returned pointer *or* pointers derived from it *may* apply the access property.

-  The "association" is *not* part of the value representation of the pointer.
-  The compiler is allowed to drop the association; it does not have a functional consequence.
-  The association *may* hold through simple expressions, sequence of simple statements, or fully inlined function
   calls where the pointer value or C++ reference is provably unchanged; this includes offset pointers used for
   array access.
-  The association is *not* expected to hold through the ABI of an unknown function call, e.g., when the pointer is
   passed through a separately-compiled function interface, unless link-time optimizations are used.

**Constraints**

- ``Property`` is convertible to :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`.

**Preconditions**:

- If ``Property`` is :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-shared>`, then it must be valid to cast the generic pointer ``ptr`` to a pointer to the *shared memory* address space.

- If ``Property`` is one of :ref:`cuda::access_property::global <libcudacxx-extended-api-memory-access-properties-access-property-global>`, :ref:`cuda::access_property::persisting <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`, :ref:`cuda::access_property::normal <libcudacxx-extended-api-memory-access-properties-access-property-normal>`, or     :ref:`cuda::access_property::streaming <libcudacxx-extended-api-memory-access-properties-access-property-streaming>`, then it must be valid to cast the generic pointer ``ptr`` to a pointer to the *global memory* address space.

*Note*: currently ``associate_access_property`` is ignored by nvcc and nvc++ on the host.

Example
-------

.. code:: cuda

    #include <cuda/cooperative_groups.h>

    __global__ void memcpy_kernel(const int* in, int* out) {
        __shared__ int smem[N];
        auto in1          = cuda::associate_access_property(in, cuda::access_property::streaming{});
        auto idx          = cooperative_groups::this_grid().thread_rank();
        smem[threadIdx.x] = in1[idx]; // streaming access
        // compute...
    }

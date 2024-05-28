.. _libcudacxx-extended-api-memory-access-properties-associate-access-property:

cuda::associate_access_property
===================================

.. code:: cuda

   template <class T, class Property>
   __host__ __device__
   T* associate_access_property(T* ptr, Property prop);

**Preconditions**:
   - if ``Property`` is :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-shared>`
     then it must be valid to cast the generic pointer ``ptr`` to a pointer to the shared memory address space.
   - if ``Property`` is one of :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-global>`,
     :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`,
     :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-normal>`, or
     :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-streaming>`
     then it must be valid to cast the generic pointer ``ptr`` to a pointer to the global memory address space.
   - if ``Property`` is a :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`
     of “range” kind, then ``ptr`` must be in the valid range.

**Mandates**: ``Property`` is convertible to :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`.

**Effects**: no effects.

**Hint**: to associate an access property with the returned pointer, such that subsequent memory operations with the
returned pointer *or* pointers derived from it *may* apply the access property.

-  The “association” is *not* part of the value representation of the pointer.
-  The compiler is allowed to drop the association; it does not have a functional consequence.
-  The association *may* hold through simple expressions, sequence of simple statements, or fully inlined function
   calls where the pointer value or C++ reference is provably unchanged; this includes offset pointers used for
   array access.
-  The association is *not* expected to hold through the ABI of an unknown function call, e.g., when the pointer is
   passed through a separately-compiled function interface, unless link-time optimizations are used.

**Note**: currently ``associate_access_property`` is ignored by nvcc and nvc++ on the host; but this might change any time.

Example
-------

.. code:: cuda

   #include <cuda/cooperative_groups.h>
   __global__ void memcpy(int const* in_, int* out) {
       int const* in = cuda::associate_access_property(in_, cuda::access_property::streaming{});
       auto idx = cooperative_groups::this_grid().thread_rank();

       __shared__ int shmem[N];
       shmem[threadIdx.x] = in[idx]; // streaming access

       // compute...
   }

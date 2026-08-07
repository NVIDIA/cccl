.. _libcudacxx-extended-api-memory-ranges_overlap:

``cuda::ranges_overlap``
========================

Defined in the ``<cuda/memory>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   bool ranges_overlap(T lhs_start, T lhs_end, T rhs_start, T rhs_end) noexcept;

   } // namespace cuda

Returns ``true`` when the half-open byte ranges ``[lhs_start, lhs_end)`` and ``[rhs_start, rhs_end)`` intersect.

**Constraints**

- ``T`` must be a forward iterator.

**Parameters**

- ``lhs_start``: The beginning of the first range.
- ``lhs_end``: The end of the first range.
- ``rhs_start``: The beginning of the second range.
- ``rhs_end``: The end of the second range.

**Return value**

- ``true`` when the two ranges overlap, ``false`` otherwise.

**Performance considerations**

- The function is optimized when the ranges are contiguous and random access iterators.

Example
-------

.. code:: cuda

    #include <cuda/memory>
    #include <cuda/std/cassert>

    __global__ void overlap_kernel() {
        int arrayA[10];
        int arrayB[10];
        assert(cuda::ranges_overlap(arrayA + 2, arrayA + 7, arrayA, arrayA + 10)); // overlap
        assert(!cuda::ranges_overlap(arrayA, arrayA + 10, arrayB, arrayB + 10));   // no overlap
    }

    int main() {
        overlap_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/nasnWz9Tv>`__

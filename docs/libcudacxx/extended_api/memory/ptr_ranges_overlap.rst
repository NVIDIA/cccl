.. _libcudacxx-extended-api-memory-ptr_ranges_overlap:

``cuda::ptr_ranges_overlap``
============================

Defined in ``<cuda/memory>`` header.

.. code:: cuda

   namespace cuda {

   [[nodiscard]] __host__ __device__ constexpr
   bool ptr_ranges_overlap(const void* ptr1_start, const void* ptr1_end,
                           const void* ptr2_start, const void* ptr2_end) noexcept

   } // namespace cuda

Returns ``true`` when the half-open byte ranges ``[ptr1_start, ptr1_end)`` and ``[ptr2_start, ptr2_end)`` intersect.

**Parameters**

- ``ptr1_start``: Pointer to the beginning of the first range.
- ``ptr1_end``: Pointer to the end of the first range.
- ``ptr2_start``: Pointer to the beginning of the second range.
- ``ptr2_end``: Pointer to the end of the second range.

**Return value**

- ``true`` when the two ranges overlap, ``false`` otherwise.

Example
-------

.. code:: cuda

    #include <cuda/memory>
    #include <cuda/std/cassert>

    __global__ void overlap_kernel() {
        int arrayA[10];
        int arrayB[10];
        assert(cuda::ptr_ranges_overlap(arrayA + 2, arrayA + 7, arrayA, arrayA + 10)); // overlap
        assert(!cuda::ptr_ranges_overlap(arrayA, arrayA + 10, arrayB, arrayB + 10)); // no overlap
    }

    int main() {
        overlap_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/GPc4T4h7x>`_

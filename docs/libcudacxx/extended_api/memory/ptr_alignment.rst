.. _libcudacxx-extended-api-memory-ptr_alignment:

``cuda::ptr_alignment``
=======================

Defined in the header ``<cuda/memory>``.

.. code:: cuda

   namespace cuda {

   [[nodiscard]] __host__ __device__ inline
   size_t ptr_alignment(const void* ptr) noexcept;

   [[nodiscard]] __host__ __device__ inline
   size_t ptr_alignment(const void* ptr, size_t max_alignment) noexcept;

   } // namespace cuda

The function returns the alignment of a pointer, namely the largest power of two that divides the pointer address,
optionally capped by ``max_alignment``.

**Parameters**

- ``ptr``: The pointer.
- ``max_alignment``: The maximum alignment to consider.

**Return value**

- The alignment of the pointer as a ``size_t`` value (always a power of two).

**Constraints**

- ``ptr`` must not be null.
- ``max_alignment`` must be a power of two.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    __global__ void kernel(const void* ptr, const void* ptr1, const void* ptr16) {
        assert(cuda::ptr_alignment(ptr)      >= 128); // usually true for cudaMalloc pointers
        assert(cuda::ptr_alignment(ptr1, 1)  == 1);
        assert(cuda::ptr_alignment(ptr16)    == 16);
        assert(cuda::ptr_alignment(ptr16, 4) == 4);
    }

    int main() {
        void* ptr;
        cudaMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr, ptr + 1, ptr + 16);
        cudaDeviceSynchronize();
        return 0;
    }

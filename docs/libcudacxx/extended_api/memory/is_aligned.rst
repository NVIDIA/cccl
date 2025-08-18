.. _libcudacxx-extended-api-memory-is_aligned:

``cuda::is_aligned``
====================

.. code:: cuda

   [[nodiscard]] __host__ __device__ inline
   bool is_aligned(const void* ptr, size_t alignment) noexcept

The function determines if a pointer is aligned to a specific alignment.

**Parameters**

- ``ptr``: The pointer.
- ``alignment``: The alignment.

**Return value**

- ``true`` if the pointer is aligned to the specified alignment, ``false`` otherwise.

**Constraints**

- ``alignment`` must be a power of two.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    __global__ void kernel(const void* ptr) {
        assert(cuda::is_aligned(ptr, 16));
    }

    int main() {
        void* ptr;
        cudaMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr);
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/K3oMTqbxa>`_

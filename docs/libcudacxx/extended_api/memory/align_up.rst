.. _libcudacxx-extended-api-memory-align_up:

``cuda::align_up``
==================

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ inline
   T* align_up(T* ptr, size_t alignment) noexcept

The function returns the original pointer or closest pointer larger than ``ptr`` that is aligned to the specified alignment :math:`ceil\left(\frac{ptr}{alignment}\right) * alignment`.

**Parameters**

- ``ptr``: The pointer.
- ``alignment``: The alignment.

**Return value**

- The original pointer or closest pointer larger than ``ptr`` that is aligned to the specified alignment.

**Constraints**

- ``alignment`` must be a power of two.
- ``alignment >= alignof(T)``.
- ``ptr`` is aligned to ``alignof(T)``.

**Performance considerations**

- The function is optimized for compile-time values of ``alignment``.
- The returned pointer is decorated with ``__builtin_assume_aligned`` to help the compiler generate better code.
- The returned pointer maintains the same memory space, for example shared memory, as the input pointer.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    __global__ void kernel(const int* ptr) {
        auto ptr_align16 = cuda::align_up(ptr, 16);
        reinterpret_cast<int4*>(ptr_align16)[0] = int4{1, 2, 3, 4};
    }

    int main() {
        int* ptr;
        cudaMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr);
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/8sYxETbjM>`_

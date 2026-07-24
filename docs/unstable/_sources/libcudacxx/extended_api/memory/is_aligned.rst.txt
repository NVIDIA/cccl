.. _libcudacxx-extended-api-memory-is_aligned:

``cuda::is_aligned``
====================

Defined in the header ``<cuda/memory>``.

.. code:: cuda

   namespace cuda {

   [[nodiscard]] __host__ __device__ inline
   bool is_aligned(const void* ptr, size_t alignment) noexcept

   } // namespace cuda

The function determines if a pointer is aligned to a specific alignment.

**Parameters**

- ``ptr``: The pointer.
- ``alignment``: The alignment.

**Return value**

- ``true`` if the pointer is aligned to the specified alignment, ``false`` otherwise.

**Constraints**

- ``alignment`` must be a power of two.

.. note::

  The function is similar to the C++ standard library function `cuda::std::is_sufficiently_aligned() <https://en.cppreference.com/w/cpp/memory/is_sufficiently_aligned.html>`__ from the ``<cuda/std/memory>`` header. The differences are the following:

  - ``cuda::is_aligned()`` doesn't have a template parameter and might be less expensive to compile.
  - ``cuda::is_aligned()`` supports run-time values of ``alignment``.
  - ``cuda::std::is_sufficiently_aligned()`` additionally checks the compatibility between the alignment of the pointer type and the specified alignment.

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

`See it on Godbolt 🔗 <https://godbolt.org/z/Tr45EoKsT>`__

.. _libcudacxx-extended-api-memory-ptr_rebind:

``cuda::ptr_rebind``
====================

Defined in the header ``<cuda/memory>``.

.. code:: cuda

    namespace cuda {

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    U* ptr_rebind(T* ptr) noexcept;

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    const U* ptr_rebind(const T* ptr) noexcept;

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    volatile U* ptr_rebind(volatile T* ptr) noexcept;

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    const volatile U* ptr_rebind(const volatile T* ptr) noexcept;

    } // namespace cuda

The functions return the pointer ``ptr`` cast to type ``U*`` or ``const U*``. They are shorter and safer alternative to ``reinterpret_cast``.

**Parameters**

- ``ptr``: The pointer.

**Return value**

- The pointer cast to type ``U*`` or ``const U*``.

**Constraints**

- ``ptr`` must be aligned to ``alignof(U)`` and ``alignof(T)``.

**Performance considerations**

- The returned pointer is decorated with ``__builtin_assume_aligned`` with the ``alignof(U)`` value to help the compiler generate better code.
- The returned pointer maintains the same memory space, for example shared memory, as the input pointer.

Example
-------

.. code:: cuda

    #include <cuda/memory>
    #include <cuda/std/cstdint>

    __global__ void kernel(const int* ptr, volatile int* ptr2) {
        auto ptr_res1 = cuda::ptr_rebind<uint64_t>(ptr);  // ptr_res1: const uint64_t*
        auto ptr_res2 = cuda::ptr_rebind<uint64_t>(ptr2); // ptr_res2: volatile uint64_t*
    }

    int main() {
        int* ptr;
        cudaMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr);
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/bavzabce9>`__

.. _libcudacxx-extended-api-memory-ptr_rebind:

``cuda::ptr_rebind``
====================

.. code:: cuda

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__ inline
    U* ptr_rebind(T* ptr) noexcept

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__ inline
    const U* ptr_rebind(const T* ptr) noexcept

The functions return the pointer ``ptr`` cast to type ``U*`` or ``const U*``. They are shorter and safer alternative to ``reinterpret_cast``.

**Parameters**

- ``ptr``: The pointer.

**Return value**

- The pointer cast to type ``U*`` or ``const U*``.

**Preconditions**

- ``alignof(U) >= alignof(T)``.

**Constraints**

- ``ptr`` must be aligned to ``alignof(U)``.

**Performance considerations**

- The returned pointer is decorated with ``__builtin_assume_aligned`` with the ``alignof(U)`` value to help the compiler generate better code.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    __global__ void kernel(const int* ptr) {
        const uint64_t* ptr2 = cuda::ptr_rebind<uint64_t>(ptr);
        // cuda::ptr_rebind<uint16_t>(ptr);  error
    }

    int main() {
        int* ptr;
        cudaMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr);
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/K3oMTqbxa>`_

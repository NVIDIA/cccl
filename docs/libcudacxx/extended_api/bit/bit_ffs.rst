.. _libcudacxx-extended-api-bit-bit_ffs:

``cuda::bit_ffs``
=================

Defined in the ``<cuda/bit>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
   int bit_ffs(T value) noexcept;

   } // namespace cuda

The function returns one plus the index of the least significant set bit of ``value``, or ``0`` if ``value`` is zero. This matches the semantics of ``__builtin_ffs`` and CUDA's ``__ffs``.

**Parameters**

- ``value``: the unsigned integer value to scan.

**Return value**

- ``0`` if ``value`` is zero, otherwise the 1-based position of the least significant set bit.

**Constraints**

- ``T`` is an unsigned integral type.

.. note::

    Unlike ``cuda::std::countr_zero``, which returns the number of trailing zero bits, ``bit_ffs`` uses a 1-based position and is well defined for a zero input.

Example
-------

.. code:: cuda

    #include <cuda/bit>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>
    #include <cuda_runtime_api.h>

    __global__ void bit_ffs_kernel() {
        assert(cuda::bit_ffs(uint32_t{0}) == 0);
        assert(cuda::bit_ffs(uint32_t{1}) == 1);
        assert(cuda::bit_ffs(uint32_t{0b10101000}) == 4);
        assert(cuda::bit_ffs(~uint32_t{0}) == 1);
    }

    int main() {
        bit_ffs_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

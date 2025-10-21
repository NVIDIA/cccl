.. _libcudacxx-extended-api-bit-ffs:

``cuda::ffs``
=============

.. code:: cpp

   template <typename T>
   [[nodiscard]] constexpr int
   ffs(T value) noexcept;

The function finds the first (least significant) set bit in ``value`` and returns its 1-based index. If ``value`` is 0, returns 0.

**Parameters**

- ``value``: The value to search for the first set bit

**Return value**

- The 1-based index of the first set bit, or 0 if ``value`` is 0

**Constraints**

- ``T`` is an unsigned integer type

**Relationship with other functions**

- For non-zero values: ``ffs(x) == countr_zero(x) + 1``

**Performance considerations**

The function performs the following operations in device code:

- ``uint32_t``: Single ``FFS`` instruction
- ``uint64_t``: Single ``FFSLL`` instruction
- ``__uint128_t``: Two ``FFSLL`` instructions with conditional logic

On host code, the function uses compiler intrinsics when available:

- GCC/Clang: ``__builtin_ffs`` / ``__builtin_ffsll``
- MSVC: ``_BitScanForward`` / ``_BitScanForward64``

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>

    __global__ void ffs_kernel() {
        assert(cuda::ffs(0u) == 0);
        assert(cuda::ffs(1u) == 1);
        assert(cuda::ffs(2u) == 2);
        assert(cuda::ffs(3u) == 1);
        assert(cuda::ffs(4u) == 3);
        assert(cuda::ffs(8u) == 4);
        assert(cuda::ffs(128u) == 8);
        assert(cuda::ffs(0x80000000u) == 32);
    }

    int main() {
        ffs_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

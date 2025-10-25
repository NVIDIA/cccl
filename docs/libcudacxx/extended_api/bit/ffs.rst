.. _libcudacxx-extended-api-bit-ffs:

``cuda::ffs``
=============

Defined in the ``<cuda/bit>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   int ffs(T value) noexcept;

   } // namespace cuda

The function finds the first (least significant) set bit in ``value`` and returns its 1-based index. If ``value`` is 0, returns 0.

**Parameters**

- ``value``: Input value

**Return value**

- The 1-based index of the first set bit, or 0 if ``value`` is 0

**Constraints**

- ``T`` must be an unsigned integer type. Supported types include all standard unsigned integer types and ``__uint128_t`` when available.

**Relationship with other functions**

- For non-zero values: ``ffs(x) == countr_zero(x) + 1``

**Performance considerations**

The function performs the following operations:

- Device:

  - ``uint8_t``, ``uint16_t``, ``uint32_t``: ``BREV``, ``FLO``, ``IADD3``

- Host:

  - GCC/Clang: ``__builtin_ffs`` / ``__builtin_ffsll``
  - MSVC: ``_BitScanForward`` / ``_BitScanForward64``
  - Other: Portable constexpr loop implementation

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void ffs_kernel() {
        assert(cuda::ffs(0u) == 0);
        assert(cuda::ffs(1u) == 1);
        assert(cuda::ffs(0b1100u) == 3);
        assert(cuda::ffs(0x80000000u) == 32);
    }

    int main() {
        ffs_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

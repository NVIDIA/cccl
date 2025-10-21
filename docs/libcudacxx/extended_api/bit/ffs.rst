.. _libcudacxx-extended-api-bit-ffs:

``cuda::ffs``
=============

.. code:: cpp

   template <typename T>
   [[nodiscard]] constexpr int
   ffs(T value) noexcept;

The function finds the first (least significant) set bit in ``value`` and returns its 1-based index. If ``value`` is 0, returns 0.

**Parameters**

- ``value``: Input value

**Return value**

- The 1-based index of the first set bit, or 0 if ``value`` is 0

**Constraints**

- ``T`` is an unsigned integer type.

**Relationship with other functions**

- For non-zero values: ``ffs(x) == countr_zero(x) + 1``

**Performance considerations**

The function performs the following operations:

- Device:

  - ``uint8_t``, ``uint16_t``, ``uint32_t``: ``FFS``
  - ``uint64_t``: ``FFSLL``
  - ``uint128_t``: ``FFSLL`` x2 with conditional logic

- Host:

  - GCC/Clang: ``__builtin_ffs`` / ``__builtin_ffsll``
  - MSVC: ``_BitScanForward`` / ``_BitScanForward64``
  - Other: Portable constexpr loop implementation

.. note::

    The function is guaranteed to be ``constexpr`` on all platforms, allowing compile-time evaluation when the input is a constant expression.

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

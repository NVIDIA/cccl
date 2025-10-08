.. _libcudacxx-extended-api-math-multiply-high-half:

``cuda::multiply_half_high``
============================

Defined in ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] constexpr
   T multiply_half_high(T lhs, T rhs) noexcept;

   } // namespace cuda

Computes the most significant half of the bits of the product of two non-negative integers ``lhs`` and ``rhs``.

**Parameters**

- ``lhs``: First multiplicand.
- ``rhs``: Second multiplicand.

**Return value**

Upper half of ``lhs * rhs`` returned as ``T``.

**Constraints**

- ``T`` is an integer type.

**Preconditions**

- ``lhs >= 0`` when ``T`` is signed.
- ``rhs >= 0`` when ``T`` is signed.

**Remarks**

- Uses ``__umulhi`` or ``__umul64hi`` instructions on device when available.
- Uses a double-width intermediate type when possible.
- Relies on a manual decomposition fallback when 128-bit intermediates are unavailable.

Example
-------

.. code:: cuda

   #include <cuda/cmath>
   #include <cuda/std/cassert>
   #include <cuda/std/cstdint>

   __global__ void multiply_high_kernel()
   {
       uint32_t lhs       = 0xABCD1234;
       uint32_t rhs       = 1 << 16; // 2^16
       uint32_t high_half = cuda::multiply_half_high(lhs, rhs);
       assert(high_half == 0xAB);
   }

   int main()
   {
       multiply_high_kernel<<<1, 1>>>();
       cudaDeviceSynchronize();
       return 0;
   }

`See it on Godbolt 🔗 <https://godbolt.org/z/dPPzsEaGM>`_

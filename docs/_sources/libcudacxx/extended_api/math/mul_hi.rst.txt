.. _libcudacxx-extended-api-math-mul-hi:

``cuda::mul_hi``
================

Defined in ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T mul_hi(T lhs, T rhs) noexcept;

   } // namespace cuda

Computes the most significant half of the bits of the product of two non-negative integers ``lhs`` and ``rhs``.

**Parameters**

- ``lhs``: First multiplicand.
- ``rhs``: Second multiplicand.

**Return value**

- The most significant half of ``lhs * rhs`` returned as ``T``.

**Constraints**

- ``T`` is an integer type.

**Remarks**

- Uses ``__mulhi``, ``__umulhi``, ``__mul64hi``, ``__umul64hi`` intrinsics on device when available.
- Uses ``__mulh``, ``__umulh`` intrinsics on Windows host code when available.
- Uses a double-width intermediate type when possible.
- Relies on a manual decomposition fallback when 128-bit intermediates are unavailable for 64-bit integers.

Example
-------

.. code:: cuda

   #include <cuda/cmath>
   #include <cuda/std/cassert>
   #include <cuda/std/cstdint>

   __global__ void mul_hi_kernel()
   {
       uint32_t lhs       = 0xABCD1234;
       uint32_t rhs       = 1 << 16; // 2^16
       uint32_t high_half = cuda::mul_hi(lhs, rhs);
       assert(high_half == 0xAB);
   }

   int main()
   {
       mul_hi_kernel<<<1, 1>>>();
       cudaDeviceSynchronize();
       return 0;
   }

`See it on Godbolt 🔗 <https://godbolt.org/z/rfb4s76nK>`_

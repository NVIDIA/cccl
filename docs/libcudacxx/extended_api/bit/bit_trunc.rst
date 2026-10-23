.. _libcudacxx-extended-api-bit-bit_trunc:

``cuda::bit_trunc``
===================

Defined in the ``<cuda/bit>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T bit_trunc(T value, int width) noexcept;

   } // namespace cuda

The function retains the ``width`` least significant bits of ``value`` and clears all remaining bits.
The result has the same type as ``value``.

**Parameters**

- ``value``: The unsigned integer value to truncate.
- ``width``: The number of least significant bits to retain.

**Return value**

- ``value`` with all bits at positions greater than or equal to ``width`` cleared.
- Zero if ``width == 0``.
- ``value`` if ``width >= cuda::std::numeric_limits<T>::digits``.

**Constraints**

- ``T`` is an unsigned integer type.

**Preconditions**

- ``width >= 0``. There is no upper bound on ``width``.

**Performance considerations**

The function uses the following instructions for runtime inputs:

- Device (SM70 and newer): ``SGXT.U32`` per 32-bit word. 64- and 128-bit types require additional operations for width clamping.
- Host (x86-64 with BMI2): ``BZHI``; two for 128-bit types and some additional operations for width clamping.
- Other cases: shift and bitwise AND.

Example
-------

.. code:: cuda

  #include <cuda/bit>
  #include <cuda/std/cassert>
  #include <cuda/std/cstdint>

  static_assert(cuda::bit_trunc(0b11001001u, 4) == 0b1001u);
  static_assert(cuda::bit_trunc(0b11001001u, 0) == 0u);
  static_assert(cuda::bit_trunc(cuda::std::uint8_t{0xff}, 100) == 0xff);

  __global__ void bit_trunc_kernel(cuda::std::uint32_t value, int width)
  {
    assert(cuda::bit_trunc(value, width) == 0b1001u);
  }

  int main()
  {
    bit_trunc_kernel<<<1, 1>>>(0b11001001u, 4);
    cudaDeviceSynchronize();
  }

`See it on Godbolt 🔗 <https://godbolt.org/z/7v44a9T8j>`__

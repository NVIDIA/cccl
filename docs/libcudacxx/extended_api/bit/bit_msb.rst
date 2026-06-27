.. _libcudacxx-extended-api-bit-bit_msb:

``cuda::bit_msb``
=================

Defined in the ``<cuda/bit>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   int bit_msb(T value) noexcept;

   } // namespace cuda

The function returns the zero-based index of the most significant set bit of ``value`` (that is, ``floor(log2(value))``), or ``-1`` if ``value`` is zero. It is the most-significant counterpart to :ref:`bit_ffs <libcudacxx-extended-api-bit-bit_ffs>` (find first set).

**Parameters**

- ``value``: the unsigned integer value to scan.

**Return value**

- ``-1`` if ``value`` is zero, otherwise the zero-based position of the most significant set bit.

**Constraints**

- ``T`` is an unsigned integral type.

.. note::

    For a non-zero ``value``, ``bit_msb(value)`` equals ``cuda::std::bit_width(value) - 1``. It is provided as a safe, type-generic way to get the most significant bit index on all supported integer types.

Example
-------

.. code:: cuda

    #include <cuda/bit>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>

    __global__ void bit_msb_kernel() {
        assert(cuda::bit_msb(uint32_t{0}) == -1);
        assert(cuda::bit_msb(uint32_t{1}) == 0);
        assert(cuda::bit_msb(uint32_t{0b10101000}) == 7);
        assert(cuda::bit_msb(~uint32_t{0}) == 31);
    }

    int main() {
        bit_msb_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

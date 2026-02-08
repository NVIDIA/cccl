.. _libcudacxx-extended-api-bit-bit_reverse:

``cuda::bit_reverse``
=====================

Defined in the ``<cuda/bit>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T bit_reverse(T value) noexcept;

   } // namespace cuda

The function reverses the order of bits in a value.

**Parameters**

- ``value``: Input value

**Return value**

- Value with reversed bits

**Constraints**

- ``T`` is an unsigned integer type.

**Performance considerations**

The function performs the following operations:

- Device:

  - ``uint8_t`` ``uint16_t``: ``PRMT``, ``BREV``
  - ``uint32_t``: ``BREV``
  - ``uint64_t``: ``BREV`` x2, ``MOV`` x2
  - ``uint128_t``: ``BREV`` x4, ``MOV`` x4

- Host: ``__builtin_bitreverse<N>`` with clang

.. note::

    Using the function could not be optimal when the input is a run-time value that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations.

.. note::

    GCC <= 8 uses a slow path with more instructions even in device code.

Example
-------

.. code:: cuda

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bit_reverse_kernel() {
        assert(cuda::bit_reverse(0x0000FFFFu) == 0xFFFF0000u);
        assert(cuda::bit_reverse(uint8_t{0b00001011}) == uint8_t{0b11010000});
    }

    int main() {
        bit_reverse_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/nW6qe5fT4>`__

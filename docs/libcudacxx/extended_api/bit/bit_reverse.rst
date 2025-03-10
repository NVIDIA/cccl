.. _libcudacxx-extended-api-bit-bit_reverse:

``cuda::bit_reverse``
=====================

.. code:: cpp

   template <typename T>
   [[nodiscard]] constexpr T
   bit_reverse(T value) noexcept;

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

    When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the function could not be optimal.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bit_reverse_kernel() {
        assert(bitfield_reverse(0u) == ~0u);
        assert(bitfield_reverse(uint8_t{0b00001011}) == uint8_t{0b11010000});
    }

    int main() {
        bit_reverse_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/K36dvoh58>`_

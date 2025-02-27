.. _libcudacxx-extended-api-bit-bit_reverse:

``bit_reverse``
===============

.. code:: cpp

   template <typename T>
   [[nodiscard]] constexpr T
   bit_reverse(T value) noexcept;

The function reverses the order of bits in a value.

**Parameters**

- ``value``: Input value

**Return value**

- Value with reversed bits

**Preconditions**

- *Compile-time*: ``T`` is an unsigned integral type (including 128-bit integers).

**Performance considerations**

The function performs the following operations:

- Device: ``BREV``
- Host: ``__builtin_bitreverse<N>`` with clang

.. note::

    When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the functions could not be optimal.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bit_reverse_kernel() {
        assert(bitfield_reverse(0u) == ~0u);
        assert(bitfield_reverse(~0u) == 0);
    }

    int main() {
        bit_reverse_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/bjM8Tjr7d>`_

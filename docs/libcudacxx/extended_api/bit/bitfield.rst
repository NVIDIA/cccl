.. _libcudacxx-extended-api-bit-bitfield:

``bitfield_insert`` and ``bitfield_extract``
============================================

.. code:: cpp

   template <typename T>
   [[nodiscard]] constexpr T
   bitfield_insert(T value, int start, int width = 1) noexcept;

   template <typename T>
   [[nodiscard]] constexpr T
   bitfield_extract(T value, int start, int width = 1) noexcept;

The functions insert or extract a bitfield from a value.
``bitfield_insert`` computes ``value | bitfield``, while ``bitfield_extract`` computes ``value & bitfield``.
``bitfield`` is a sequence of bit of width ``width`` shifted left by ``start``.

**Parameters**

- ``value``: The value to apply the bitfield.
- ``start``:  Initial position of the bitfield.
- ``width``:  Width of the bitfield.

**Return value**

- ``bitfield_insert``: ``value | bitfield``.
- ``bitfield_extract``: ``value & bitfield``.

**Preconditions**

- *Compile-time*: ``T`` is an unsigned integral type (including 128-bit integers).
- *Run-time* (debug mode):

    - ``start >= 0 && start < num_bits(T)``
    - ``width >  0 && width <= num_bits(T)``
    - ``start + width <= num_bits(T)``

**Performance considerations**

The functions perform the following operations in CUDA:

- ``SM < 70``: ``BFI/BFE`` + bitwise and/or
- ``SM >= 70``: ``BMSK`` + bitwise and/or

.. note::

    When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the functions could not be optimal.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bitfield_kernel() {
        assert(bitfield_insert(0u, 0, 4) == 0b1111);
        assert(bitfield_insert(0u, 3, 4) == 0b1111000);
        assert(bitfield_insert(1u, 3, 4) == 0b1111001);

        assert(bitfield_extract(~0u, 0, 4) == 0b1111);
        assert(bitfield_extract(~0u, 3, 4) == 0b1111000);
        assert(bitfield_extract(0b00100111u, 3, 4) == 0b00100000);
    }

    int main() {
        bitfield_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/j7W5hjrfa>`_

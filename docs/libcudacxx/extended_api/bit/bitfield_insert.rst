.. _libcudacxx-extended-api-bit-bitfield_insert:

``cuda::bitfield_insert``
=========================

.. code:: cpp

   template <typename T>
   [[nodiscard]] constexpr T
   bitfield_insert(T dest, T source, int start, int width) noexcept;

The function extracts the lower bitfield of size ``width`` from ``source`` and inserts it into ``dest`` at position ``start``.

**Parameters**

- ``dest``:   The value to insert the bitfield.
- ``source``: The value from which extract the bitfield.
- ``start``:  Initial position of the bitfield.
- ``width``:  Width of the bitfield.

**Return value**

- ``((value << start) & mask) | (source & ~mask)``, where ``mask`` is a bitmask of width ``width``.

**Constraints**

- ``T`` is an unsigned integer type.

**Preconditions**

    - ``start >= 0 && start <= num_bits(T)``
    - ``width >= 0 && width <= num_bits(T)``
    - ``start + width <= num_bits(T)``

**Performance considerations**

The function performs the following operations in CUDA for ``uint8_t``, ``uint16_t``, ``uint32_t``:

- ``SM < 70``: ``BFI``
- ``SM >= 70``: ``BMSK``, bitwise operation x5

.. note::

    When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the function could not be optimal.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bitfield_insert_kernel() {
        assert(cuda::bitfield_insert(0u, 0xFFFFu, 0, 4) == 0b1111);
        assert(cuda::bitfield_insert(0u, 0xFFFFu, 3, 4) == 0b1111000);
        assert(cuda::bitfield_insert(1u, 0xFFFFu, 3, 4) == 0b1111001);
    }

    int main() {
        bitfield_insert_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/Phs8czqes>`_

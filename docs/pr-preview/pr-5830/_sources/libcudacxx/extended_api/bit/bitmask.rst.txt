.. _libcudacxx-extended-api-bit-bitmask:

``cuda::bitmask``
=================

.. code:: cpp

   template <typename T = uint32_t>
   [[nodiscard]] constexpr T
   bitmask(int start, int width) noexcept;

The function generates a bitmask of size ``width`` starting at position ``start``.

**Parameters**

- ``start``: starting position of the bitmask
- ``width``: width of the bitmask

**Return value**

- Bitmask of size ``width`` starting at ``start``

**Constraints**

- ``T`` is an unsigned integral type.

**Preconditions**

    - ``start >= 0 && start <= num_bits(T)``
    - ``width >= 0 && width <= num_bits(T)``
    - ``start + width <= num_bits(T)``

**Performance considerations**

The function performs the following operations in device code:

- ``uint8_t``, ``uint16_t``, ``uint32_t``: ``BMSK``
- ``uint64_t``: ``SHL`` x4, ``ADD`` x2

.. note::

    When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the function could not be optimal.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA

Example
-------

.. code:: cpp

    #include <cuda/bit>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>

    __global__ void bitmask_kernel() {
        assert(cuda::bitmask(2, 4) == 0b111100u);
        assert(cuda::bitmask<uint64_t>(1, 3) == uint64_t{0b1110});
    }

    int main() {
        bitmask_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/PPqP8rTPd>`_

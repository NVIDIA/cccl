.. _libcudacxx-extended-api-bit-bitfield_extract:

``cuda::bitfield_extract``
==========================

Defined in the ``<cuda/bit>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T bitfield_extract(T value, int start, int width) noexcept;

   } // namespace cuda

The function extracts a bitfield from a value and returns it in the lower bits.
``bitfield_extract()`` computes ``(value >> start) & mask``, where ``mask`` is a sequence of bits of width ``width``.

**Parameters**

- ``value``: The value to apply the bitfield.
- ``start``:  Initial position of the bitfield.
- ``width``:  Width of the bitfield.

**Return value**

- ``(value >> start) & mask``, where ``mask`` is a bitmask of width ``width``.

**Constraints**

- ``T`` is an unsigned integer type.

**Preconditions**

- ``start >= 0 && start <= num_bits(T)``.
- ``width >= 0 && width <= num_bits(T)``.
- ``start + width <= num_bits(T)``.

**Performance considerations**

The function performs the following operations in CUDA for ``uint8_t``, ``uint16_t``, ``uint32_t``:

- ``SM < 70``: ``BFE``.
- ``SM >= 70``: ``BMSK``, bitwise operation x2.

.. note::

    Using the function could not be optimal when the input is a run-time value that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA.

Example
-------

.. code:: cuda

    #include <cuda/bit>
    #include <cuda/std/cassert>

    __global__ void bitfield_extract_kernel() {
        assert(cuda::bitfield_extract(~0u, 0, 4) == 0b1111);
        assert(cuda::bitfield_extract(0b1011000u, 3, 4) == 0b1011);
    }

    int main() {
        bitfield_extract_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/ofGnbsxW5>`__

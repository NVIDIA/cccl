.. _libcudacxx-extended-api-math-pow2:

Power of Two Utilities
======================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   bool is_power_of_two(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T next_power_of_two(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   T prev_power_of_two(T value) noexcept;

   } // namespace cuda

The functions provide utilities to determine if an integer value is a power of two, and to compute the next and previous power of two.

**Parameters**

- ``value``: The input value.

**Return value**

- ``is_power_of_two``: Return ``true`` if ``value`` is a power of two, ``false`` otherwise.
- ``next_power_of_two``: Return the smallest power of two greater than or equal to ``value``.
- ``prev_power_of_two``: Return the largest power of two less than or equal to ``value``.

**Constraints**

- ``T`` is an integer types. Contrary to ``cuda::std::has_single_bit``, ``cuda::std::bit_floor``, and ``cuda::std::bit_ceil``, ``T`` can be both signed and unsigned.

**Preconditions**

- ``value > 0``

**Performance considerations**

See :ref:`\<cuda/std/bit\> performance considerations <libcudacxx-standard-api-numerics-bit>`

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void pow2_kernel() {
        assert(!cuda::is_power_of_two(20));
        assert(cuda::next_power_of_two(20) == 32);
        assert(cuda::prev_power_of_two(20) == 16);
    }

    int main() {
        pow2_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/896Yx3vf8>`__

.. _libcudacxx-extended-api-math-check-overflow:

Check Overflows
===============

.. code:: cuda

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline
   constexpr bool is_add_overflow(T a, U b) noexcept;

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline
   constexpr bool is_sub_overflow(T a, U b) noexcept;

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline
   constexpr bool is_mul_overflow(T a, U b) noexcept;

The function checks is addition, subtraction, or multiplication of two integrals (including 128-bit integers) overflows the maximum value or underflow the minimum value of the common type (``cuda::std::common_type_t<T, U>``).

**Parameters**

- ``a``: First value.
- ``b``: Second value.

**Return value**

``True`` if the addition, subtraction, or multiplication results in an overflow/underflow. ``false`` otherwise.

**Performance considerations**

- All functions directly return ``false`` for all data type combinations that do not result in an overflow/underflow.
- Checking for overflows/underflows is more expensive when the common type is a ``signed`` integer.
- Checking for overflows/underflows for multiplication is very expensive especially on device code because it involves a division.

Example
-------

.. code:: cuda

    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>
    #include <cuda/std/limits>

    int main() {
        constexpr auto int32_max = cuda::std::numeric_limits<int>::max();
        constexpr auto int32_min = cuda::std::numeric_limits<int>::min();
        constexpr auto int8_max  = cuda::std::numeric_limits<int8_t>::max();
        constexpr auto uint8_max = cuda::std::numeric_limits<uint8_t>::max();

        assert(cuda::is_add_overflow(int32_max, 1));
        assert(cuda::is_add_overflow(int8_max, int8_t{1}));
        assert(not cuda::is_add_overflow(int8_max, 1));
        assert(not cuda::is_add_overflow(int8_max, uint8_max));

        assert(not cuda::is_add_overflow(-int32_max, -1));
        assert(cuda::is_add_overflow(int32_min, -1));

        assert(cuda::is_sub_overflow(int32_min, 1));
        assert(not cuda::is_sub_overflow(int32_min, 0));
        assert(cuda::is_sub_overflow(0, int32_min)); // negation overflow

        assert(cuda::is_mul_overflow(uint8_t{16}, uint8_t{16}));   // 256
        assert(cuda::is_mul_overflow(int8_t{16}, int8_t{8}));      // 128
        assert(not cuda::is_mul_overflow(int8_t{-16}, int8_t{8})); // -128
    }

`See it on Godbolt 🔗 <>`_

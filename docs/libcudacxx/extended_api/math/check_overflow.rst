.. _libcudacxx-extended-api-math-check-overflow:

Check Overflow
==============

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

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline
   constexpr bool is_div_overflow(T a, U b) noexcept;

The function checks if addition, subtraction, multiplication, or division of two integrals (including 128-bit integers) overflows the maximum value or underflow the minimum value of the common type (``cuda::std::common_type_t<T, U>``).

.. note::

    ``cuda::std::common_type_t<T, U>`` behaves as implicit integer promotions expect in the case of ``T == U``, where the resulting type is maintained (``T/U``).

**Parameters**

- ``a``: First value.
- ``b``: Second value.

**Return value**

``True`` if the addition, subtraction, multiplication, or division results in an overflow/underflow. ``false`` otherwise.

*Note:* ``is_div_overflow()`` treats ``b == 0`` as an overflow.

**Performance considerations**

- All functions directly return ``false`` for all data type combinations that do not result in an overflow/underflow.
- Checking for overflows/underflows is more expensive when the common type is a ``signed`` integer.
- Checking for overflows/underflows for multiplication is very expensive especially on device code because it involves a division.

Example
-------

.. code:: cuda

    #include <cuda/std/cassert>
    #include <cuda/std/cmath>
    #include <cuda/std/cstdint>
    #include <cuda/std/limits>

    int main() {
        constexpr auto int32_max = cuda::std::numeric_limits<int>::max();
        constexpr auto int32_min = cuda::std::numeric_limits<int>::min();
        constexpr auto int8_max  = cuda::std::numeric_limits<int8_t>::max();
        constexpr auto uint8_max = cuda::std::numeric_limits<uint8_t>::max();

        // add examples
        assert(cuda::is_add_overflow(int32_max, 1));
        assert(cuda::is_add_overflow(int8_max, int8_t{1}));
        assert(not cuda::is_add_overflow(int8_max, 1));         // promotion
        assert(not cuda::is_add_overflow(int8_max, uint8_max)); // promotion

        assert(not cuda::is_add_overflow(-int32_max, -1));      // -min
        assert(cuda::is_add_overflow(int32_min, -1));

        // subtraction examples
        assert(cuda::is_sub_overflow(int32_min, 1));
        assert(cuda::is_sub_overflow(0, int32_min)); // negation overflow

        // multiplication examples
        assert(cuda::is_mul_overflow(uint8_t{16}, uint8_t{16}));   // 256
        assert(cuda::is_mul_overflow(int8_t{16}, int8_t{8}));      // 128
        assert(not cuda::is_mul_overflow(int8_t{-16}, int8_t{8})); // -128

        assert(cuda::is_mul_overflow(int32_min, -1));
        assert(not cuda::is_mul_overflow(int32_min + 1, -1)); // max
        assert(not cuda::is_mul_overflow(int32_max, -1));
        assert(cuda::is_mul_overflow(int32_max, -2));

        // division examples
        assert(not cuda::is_div_overflow(8, 4));
        assert(cuda::is_div_overflow(int32_min, -1));
        assert(not cuda::is_div_overflow(int32_min, -2));
        assert(cuda::is_div_overflow(5, 0));
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/bd8114EGj>`_

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

The function checks is addition, subtraction, or multiplication of two integral (including 128-bit integers) overflows the maximum value or underflow the minimum value of the common type (``cuda::std::common_type_t<T, U>``).

**Parameters**

- ``a``: First value.
- ``b``: Second value.

**Return value**

``True`` if the addition, subtraction, or multiplication results in an overflow/underflow. ``false`` otherwise.

**Performance considerations**

- All functions directly return ``false`` for all data type combinations that does not result in an overflow/underflow.
- Checking for overflows/underflows is more expensive when the common type is a ``signed`` integer.
- Checking for overflows/underflows for multiplication is very expensive especially on device code because it involves a division.

Example
-------

.. code:: cuda

    #include <cuda/cmath>

    int main() {
        return 0;
    }

`See it on Godbolt 🔗 <>`_

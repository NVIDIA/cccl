.. _libcudacxx-extended-api-numeric-sub_overflow:

``cuda::sub_overflow``
==========================

.. code:: cpp

   template <class T>
   struct overflow_result;

   template <class Result = /*unspecified*/, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ inline constexpr
   overflow_result</*see-below*/> sub_overflow(Lhs lhs, Rhs rhs) noexcept; // (1)

   template <class Result, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ inline constexpr
   bool sub_overflow(Result& result, Lhs lhs, Rhs rhs) noexcept; // (2)

The function ``cuda::sub_overflow`` performs subtraction of two values ``lhs`` and ``rhs`` with overflow checking. The result is the same as if the operands were first promoted to an infinite precision signed type, subtracted and the result truncated to the type of the return value.

**Parameters**

- ``result``: The result of the subtraction (2).
- ``lhs``: The left-hand side operand (1, 2).
- ``rhs``: The right-hand side operand (1, 2).

**Return value**

- (1) Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object that contains the result of the subtraction and a boolean indicating whether an overflow occurred. If the ``Result`` type is specified, it will be used as the type of the result, otherwise the common type of ``Lhs`` and ``Rhs`` is used.
- (2) Returns a boolean indicating whether an overflow occurred, and if no overflow occurred,
  the result is stored in the provided ``result`` variable.

**Constraints**

- ``Result``, ``Lhs``, and ``Rhs`` must be cv-unqualified integer types.

**Performance considerations**

- TODO

Example
-------

.. code:: cuda

    #include <cuda/numeric>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        constexpr auto int_max = cuda::std::numeric_limits<int>::max();
        constexpr auto int_min = cuda::std::numeric_limits<int>::min();

        // cuda::sub_overflow(lhs, rhs) returning common type of lhs and rhs
        if (auto result = cuda::sub_overflow(int_max, -1))
        {
            assert(result.value == int_min);
        }
        else
        {
            assert(false); // should not be reached
        }

        // cuda::sub_overflow<Result>(lhs, rhs) with explicit return type
        if (auto result = cuda::sub_overflow<long long>(int_min, 1))
        {
            assert(false); // should not be reached
        }
        else
        {
            assert(result.value == static_cast<long long>(int_min) - 1ll);
        }

        unsigned result{};

        // cuda::sub_overflow(result, lhs, rhs) with bool return type
        if (cuda::sub_overflow(result, int_max, -1))
        {
            assert(false); // should not be reached
        }
        assert(result.value == static_cast<unsigned>(int_max) + 1u);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }


`See it on Godbolt 🔗 <https://godbolt.org/z/joa6jTfc4>`_

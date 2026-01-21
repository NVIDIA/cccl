.. _libcudacxx-extended-api-numeric-add_overflow:

``cuda::add_overflow``
======================

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   template <class T>
   struct overflow_result;

   template <class Result = /*unspecified*/, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ inline constexpr
   overflow_result</*see-below*/> add_overflow(Lhs lhs, Rhs rhs) noexcept; // (1)

   template <class Result, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ inline constexpr
   bool add_overflow(Result& result, Lhs lhs, Rhs rhs) noexcept; // (2)

The function ``cuda::add_overflow`` performs addition of two values ``lhs`` and ``rhs`` with overflow checking. The result is the same as if the operands were first promoted to an infinite precision signed type, added together and the result truncated to the type of the return value.

**Parameters**

- ``result``: The result of the addition (2).
- ``lhs``: The left-hand side operand (1, 2).
- ``rhs``: The right-hand side operand (1, 2).

**Return value**

1. Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object that contains the result of the addition and a boolean indicating whether an overflow occurred. If the ``Result`` type is specified, it will be used as the type of the result, otherwise the common type of ``Lhs`` and ``Rhs`` is used.
2. Returns a boolean indicating whether an overflow occurred, and if no overflow occurred, the result is stored in the provided ``result`` variable.

**Constraints**

- ``Result``, ``Lhs``, and ``Rhs`` must be integer types.

**Performance considerations**

- No overflow checking if ``Lhs +  Rhs`` is always  representable with the ``Result`` type.
- The computation is faster when ``Lhs``, ``Rhs``, and ``Result`` have the same sign.
- The computation with unsigned types is faster than signed types.
- The function uses PTX ``asm`` on device, and compiler-intrinsics on host whenever possible.

Example
-------

.. code:: cuda

    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        constexpr auto int_max = cuda::std::numeric_limits<int>::max();
        constexpr auto int_min = cuda::std::numeric_limits<int>::min();

        // cuda::add_overflow(lhs, rhs) returning common type of lhs and rhs
        // 'result' is evaluated to true if an overflow occurred, false otherwise
        if (auto result = cuda::add_overflow(1, int_max))
        {
            assert(result.value == int_min);
        }

        // cuda::add_overflow<Result>(lhs, rhs) with explicit return type
        auto result = cuda::add_overflow<long long>(-1, int_min)
        assert(!result.overflow); // no overflow
        assert(result.value == static_cast<long long>(int_min) + (-1ll));

        unsigned result{};
        // cuda::add_overflow(result, lhs, rhs) with bool return type
        if (cuda::add_overflow(result, 1, int_max))
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

`See it on Godbolt 🔗 <https://godbolt.org/z/n1dWPf8c6>`_

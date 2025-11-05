.. _libcudacxx-extended-api-numeric-sub_overflow:

``cuda::sub_overflow``
======================

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   struct overflow_result;

   template <class Result = /*unspecified*/, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ constexpr
   overflow_result</*see-below*/> sub_overflow(Lhs lhs, Rhs rhs) noexcept; // (1)

   template <class Result, class Lhs, class Rhs>
   [[nodiscard]] __host__ __device__ constexpr
   bool sub_overflow(Result& result, Lhs lhs, Rhs rhs) noexcept; // (2)

   } // namespace cuda

The function ``cuda::sub_overflow`` performs subtraction of two values ``lhs`` and ``rhs`` with overflow detection. The result is the same as if the operands were first promoted to an infinite precision signed type, subtracted, and then the result truncated to the type of the return value.

**Parameters**

- ``result``: The result of the subtraction (2).
- ``lhs``: The left-hand side operand (1, 2).
- ``rhs``: The right-hand side operand (1, 2).

**Return value**

1. Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object containing the computed  difference and a boolean flag indicating whether an overflow or underflow occurred. If the ``Result`` type is specified, it will be used as the type of the result, otherwise the common type of ``Lhs`` and ``Rhs`` is used.
2. Returns ``true`` if an overflow or underflow occurred, ``false`` otherwise.

**Constraints**

- ``Result``, ``Lhs``, and ``Rhs`` must be integer types.

**Performance considerations**

- No overflow checking is required when ``lhs - rhs`` is always representable with the ``Result`` type.
- Computation is generally faster when ``Lhs``, ``Rhs``, and ``Result`` have the same signedness.
- Unsigned computations are generally faster than signed computations.
- The function uses PTX ``asm`` on device and compiler intrinsics on host whenever possible.

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

        // cuda::sub_overflow(lhs, rhs) returning the common type of lhs and rhs
        auto result = cuda::sub_overflow(3, 1))
        assert(result.value == 2);


        // cuda::sub_overflow<Result>(lhs, rhs) with an explicit result type
        auto result2 = cuda::sub_overflow<long long>(int_min, 1);
        assert(!result2.overflow);
        assert(result2.value == static_cast<long long>(int_min) - 1ll);

        unsigned result3{};
        // cuda::sub_overflow(result, lhs, rhs) returning a bool flag
        if (!cuda::sub_overflow(result, 5u, 3u))
        {
            assert(result == 2u);
        }
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/Pq8sc9s7a>`_

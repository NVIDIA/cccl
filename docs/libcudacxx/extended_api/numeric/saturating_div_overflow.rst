.. _libcudacxx-extended-api-numeric-saturating_div_overflow:

``cuda::saturating_div_overflow``
=================================

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   struct overflow_result;

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   overflow_result<T> saturating_div_overflow(T lhs, T rhs) noexcept; // (1)

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   bool saturating_div_overflow(T& result, T lhs, T rhs) noexcept; // (2)

   } // namespace cuda

The function ``cuda::saturating_div_overflow`` performs saturating integer division of ``lhs`` by ``rhs`` with overflow and error detection.

**Parameters**

- ``result``: Result of the saturating integer division. (2)
- ``lhs``: The dividend. (1, 2)
- ``rhs``: The divisor. (1, 2)

**Return value**

1. Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object containing the result of saturating integer division and a boolean flag indicating whether an overflow or underflow occurred.
2. Returns ``true`` if an overflow or underflow occurred.

**Preconditions**

- ``rhs != 0``

**Constraints**

- ``T`` must be an `integer type <https://eel.is/c++draft/basic.fundamental#1>`_.

**Performance considerations**

- Functionality is implemented by correcting the ``cuda::div_overflow`` result in case of overflow/underflow.
- Unsigned computations are generally faster than signed computations.

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

        const auto result = cuda::saturating_div_overflow(int_min, -1); // saturated
        assert(result.value == int_max);
        assert(result.overflow);

        int value;
        if (cuda::saturating_div_overflow(value, 256, 8))
        {
            assert(false); // shouldn't be reached
        }
        assert(value == 32);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/n669ETr8E>`_

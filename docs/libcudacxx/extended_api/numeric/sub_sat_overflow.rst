.. _libcudacxx-extended-api-numeric-saturating_sub_overflow:

``cuda::saturating_sub_overflow``
======================

Defined in ``<cuda/numeric>`` header.

.. code:: cpp

   namespace cuda {

   template <class T>
   struct overflow_result;

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   overflow_result<T> saturating_sub_overflow(T lhs, T rhs) noexcept; // (1)

   template <class T>
   [[nodiscard]] __host__ __device__ constexpr
   bool saturating_sub_overflow(T& result, T lhs, T rhs) noexcept; // (2)

   } // namespace cuda

The function ``cuda::saturating_sub_overflow`` performs saturating subtraction of two values ``lhs`` and ``rhs`` with overflow detection.

**Parameters**

- ``result``: The result of the saturating subtraction. (2)
- ``lhs``: The left-hand side operand. (1, 2)
- ``rhs``: The right-hand side operand. (1, 2)

**Return value**

1. Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object containing the result of the saturating subtraction and a boolean flag indicating whether an overflow or underflow occurred.
2. Returns ``true`` if an overflow or underflow occurred, ``false`` otherwise.

**Constraints**

- ``T`` must be an `integer type <https://eel.is/c++draft/basic.fundamental#1>`_.

**Performance considerations**

- Functionality is implemented by correcting the ``cuda::sub_overflow`` result in case of overflow/underflow.
- Unsigned computations are generally faster than signed computations.

Example
-------

.. code:: cuda

    #include <cuda/numeric>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void kernel()
    {
        constexpr auto uint_max = cuda::std::numeric_limits<unsigned>::max();

        const auto result = cuda::saturating_sub_overflow(42, uint_max); // saturated
        assert(result.value == 0);
        assert(result.overflow);

        int value;
        if (cuda::saturating_sub_overflow(value, 42, 1024))
        {
            assert(false); // shouldn't be reached
        }
        assert(value == 982);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/4cnTKPjbG>`_

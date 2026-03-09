.. _libcudacxx-extended-api-numeric-overflow_cast:

``cuda::overflow_cast``
==========================

.. code:: cpp

   template <class T>
   struct overflow_result;

   template <class To, class From>
   [[nodiscard]] __host__ __device__ inline constexpr
   overflow_result<To> overflow_cast(From from) noexcept;

The function ``cuda::overflow_cast`` casts a value of type ``From`` to type ``To`` with overflow checking.

**Parameters**

- ``from``: The value to be casted.

**Return value**

- Returns an :ref:`overflow_result <libcudacxx-extended-api-numeric-overflow_result>` object that contains the result of the cast and a boolean indicating whether an overflow occurred.

**Constraints**

- ``To`` and ``From`` must be `integer types <https://eel.is/c++draft/basic.fundamental#1>`_.

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

        if (auto result = cuda::overflow_cast<unsigned>(int_max))
        {
            assert(false); // Should not be reached
        }
        else
        {
            assert(result.value == static_cast<unsigned>(int_max));
        }

        if (auto result = cuda::overflow_cast<unsigned>(int_min))
        {
            assert(result.value == static_cast<unsigned>(int_min));
        }
        else
        {
            assert(false); // Should not be reached
        }
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/vK3WMMs1n>`_

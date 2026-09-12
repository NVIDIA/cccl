.. _libcudacxx-extended-api-math-uabs:

``cuda::uabs``
====================================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::make_unsigned_t<T> uabs(T value) noexcept;

   } // namespace cuda

The function computes the absolute value of the input value. The result is returned as an unsigned integer type of the same size as the input value. In comparison to the standard ``abs`` function, the ``uabs`` eliminates the undefined behaviour when a signed ``T_MIN`` is passed as an input.

**Parameters**

- ``value``: The input value.

**Return value**

- The unsigned absolute value of the input value.

**Constraints**

- ``T`` is an integer type.

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void uabs_kernel() {
        using cuda::std::numeric_limits;

        assert(cuda::uabs(20) == 20u);
        assert(cuda::uabs(-32) == 32u);
        assert(cuda::uabs(numeric_limits<int>::max()) == static_cast<unsigned>(numeric_limits<int>::max()));
        assert(cuda::uabs(numeric_limits<int>::min()) == static_cast<unsigned>(numeric_limits<int>::max()) + 1);
    }

    int main() {
        uabs_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/KEoYfq53G>`__

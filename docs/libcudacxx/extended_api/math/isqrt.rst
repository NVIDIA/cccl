.. _libcudacxx-extended-api-math-isqrt:

``cuda::isqrt``
====================================

.. code:: cpp

   template <typename T>
   [[nodiscard]] __host__ __device__ inline constexpr
   T isqrt(T value) noexcept;

The function computes the integer square root of the input value rounded down.

**Parameters**

- ``value``: The input value.

**Return value**

- The square root value of the input value rounded down.

**Constraints**

- ``T`` is an integer type.

**Preconditions**

- ``value`` is non-negative.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void isqrt_kernel() {
        assert(cuda::isqrt(1) == 1);
        assert(cuda::isqrt(4) == 2);
        assert(cuda::isqrt(42) == 6);
        assert(cuda::isqrt(99) == 9);
        assert(cuda::isqrt(100) == 10);
    }

    int main() {
        isqrt_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/enP8cW6nY>`_

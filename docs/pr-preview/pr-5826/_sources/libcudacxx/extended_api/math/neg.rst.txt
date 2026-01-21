.. _libcudacxx-extended-api-math-neg:

``cuda::neg``
====================================

.. code:: cpp

   template <typename T>
   [[nodiscard]] __host__ __device__ inline constexpr
   T neg(T value) noexcept;

The function computes the negation of the input value accepting both signed and unsigned integer types. It doesn't emit any warnings for signed integer overflow and applying ``-`` to unsigned integer types.

**Parameters**

- ``value``: The input value.

**Return value**

- The negated value of the input value.

**Constraints**

- ``T`` is an integer type.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>

    __global__ void neg_kernel() {
        using cuda::std::numeric_limits;

        assert(cuda::neg(1) == -1);
        assert(cuda::neg(20) == -20);
        assert(cuda::neg(127u) == 4294967169u);
        assert(cuda::neg(-127) == 127);
        assert(cuda::neg(cuda::std::numeric_limits<int>::min()) == cuda::std::numeric_limits<int>::min());
    }

    int main() {
        neg_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/jq6a1odz7>`_

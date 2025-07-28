.. _libcudacxx-extended-api-math-rsqrt:

Reciprocal Square Root
======================

.. code:: cuda

   namespace cuda {

   template </*floating-point-type*/ T>
   [[nodiscard]] __host__ __device__ inline
   T rsqrt(T value) noexcept; // (1)

   template </*integral-type*/ T>
   [[nodiscard]] __host__ __device__ inline
   double rsqrt(T value) noexcept; // (2)

   } // namespace cuda

The function computes the reciprocal square root of the input value.

**Parameters**

- ``value``: The input value.

**Return value**

- If the value is :math:`\infty`, returns :math:`+0.0`.
- If the value is :math:`\pm 0.0`, returns :math:`\pm \infty`.
- If the value is :math:`\lt 0.0`, returns :math:`\text{NaN}` of an unspecified sign.
- If the value is :math:`\pm \text{NaN}`, returns :math:`\text{NaN}` of an unspecified sign.
- Otherwise, returns the reciprocal square root of the value, i.e., :math:`\frac{1}{\sqrt{\text{value}}}`.

**Constraints**

- ``T`` is a floating-point or integral type.

**Performance considerations**

On device, the function maps to the ``rsqrt.approx`` PTX instruction.

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void rsqrt_kernel() {
        assert(cuda::rsqrt(4.0f) == 0.5f);
        assert(cuda::rsqrt(cuda::std::numeric_limits<double>::infinity()) == +0.0);
        assert(cuda::std::isnan(cuda::rsqrt(-16.0)));
    }

    int main() {
        rsqrt_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/48deG7b1e>`_

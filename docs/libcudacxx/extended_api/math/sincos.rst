.. _libcudacxx-extended-api-math-sincos:

``cuda::sincos``
====================================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <class T>
   struct sincos_result
   {
     T sin;
     T cos;
   };

   template </*floating-point-type*/ T>
   [[nodiscard]] __host__ __device__
   sincos_result<T> sincos(T value) noexcept; // (1)

   template <class Integral>
   [[nodiscard]] __host__ __device__
   sincos_result<double> sincos(Integral value) noexcept; // (2)

   } // namespace cuda

Computes :math:`\sin value` and :math:`\cos value` at the same time using more efficient algorithms than if operations were computed separately.

**Parameters**

- ``value``: The input value.

**Return value**

- ``cuda::sincos_result`` object with both values set to ``NaN`` if the input value is :math:`\pm\infty` or ``NaN`` and to results of :math:`\sin value` and :math:`\cos value` otherwise. (1)
- if ``T`` is an integral type, the input value is treated as ``double``. (2)

**Constraints**

- ``T`` is an arithmetic type.

**Performance considerations**

- If available, the functionality is implemented by compiler builtins, otherwise fallbacks to ``cuda::std::sin(value)`` and ``cuda::std::cos(value)``.

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void sincos_kernel() {
        auto [sin_pi, cos_pi] = cuda::sincos(0.f);
        assert(sin_pi == 0.f);
        assert(cos_pi == 1.f);
    }

    int main() {
        sincos_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/99PP9s1z6>`__

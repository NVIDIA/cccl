.. _libcudacxx-extended-api-math-ipow:

``cuda::ipow``
====================================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T, typename E>
   [[nodiscard]] __host__ __device__ constexpr
   T ipow(T base, E exp) noexcept;

   } // namespace cuda

The function computes the integer ``base`` raised to the power of ``exp``.

**Parameters**

- ``base``: The base value.
- ``exp``: The exponent value.

**Return value**

- The result of raising ``base`` to the power of ``exp``. If ``exp`` is negative, the result is 0.

**Constraints**

- ``T`` is an integer type.
- ``E`` is an integer type.

**Preconditions**

- if ``base`` is 0, then ``exp`` must be non-negative.

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void ipow_kernel() {
        assert(cuda::ipow(0, 0) == 1);
        assert(cuda::ipow(2, 2) == 4);
        assert(cuda::ipow(99, 1) == 99);
        assert(cuda::ipow(4, 7) == 16384);
        assert(cuda::ipow(-1, 3) == -1);
        assert(cuda::ipow(23, -1) == 0);
    }

    int main() {
        ipow_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/TMacWvz8v>`__

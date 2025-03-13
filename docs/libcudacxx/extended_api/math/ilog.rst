.. _libcudacxx-extended-api-math-ilog:

``cuda::ceil_div``
==================

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ inline constexpr
   int ilog2(T value) noexcept;

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ inline constexpr
   int ilog10(T value) noexcept;

The functions compute the logarithm to the base 2 and 10 respectively of an integer value.

**Parameters**

- ``value``: The input value.

**Return value**

- The logarithm to the base 2 and 10 respectively, rounded down to the nearest integer.

**Constraints**

- ``T`` is an integer types.

**Preconditions**

- ``value > 0``

**Performance considerations**

The function performs the following operations in device code:

- ``ilog2``: ``FLO``
- ``ilog10``: ``FLO``, ``FMUL``, ``FRND.CEIL``, ``F2I``, constant memory lookup

Example
-------

.. code:: cuda

    #include <cuda/cmath>

    __global__ void ilog_kernel() {
    }

    int main() {
        ilog_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/hbxscWGT9>`_

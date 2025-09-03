.. _libcudacxx-extended-api-math-ilog:

``cuda::ilog2`` and ``cuda::ilog10``
====================================

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ inline constexpr
   int ilog2(T value) noexcept;

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ inline constexpr
   int ceil_ilog2(T value) noexcept;

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ inline constexpr
   int ilog10(T value) noexcept;

The functions compute the logarithm to the base 2 and 10 of an integer value.

**Parameters**

- ``value``: The input value.

**Return value**

- ``ilog2``, ``ceil_ilog2``: The logarithm to the base 2, rounded down and up to the nearest integer respectively.
-  ``ilog10``: The logarithm to the 10, rounded down to the nearest integer.

**Constraints**

- ``T`` is an integer types.

**Preconditions**

- ``value > 0``

**Performance considerations**

The function performs the following operations in device code:

- ``ilog2``: ``FLO``
- ``ceil_ilog2``: ``FLO``, ``POPC``, ``ADD``, comparison
- ``ilog10``: ``FLO``, ``FMUL``, ``F2I``, constant memory lookup, ``SEL`` + ``IADD`` only if ``T == uint32_t`` or ``T == __uint128_t``

Example
-------

.. code:: cpp

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void ilog_kernel() {
        assert(cuda::ilog2(20) == 4);
        assert(cuda::ceil_ilog2(20) == 5);
        assert(cuda::ilog2(32) == 5);
        assert(cuda::ceil_ilog2(32) == 5);
        assert(cuda::ilog10(100) == 2);
        assert(cuda::ilog10(2000) == 3);
    }

    int main() {
        ilog_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/nqrYvrGTq>`_

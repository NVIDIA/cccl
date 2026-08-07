.. _libcudacxx-extended-api-math-ilog:

``cuda::ilog2``, ``cuda::ceil_ilog2``, ``cuda::ilog10``, and ``cuda::ceil_ilog10``
=================================================================================

Defined in the ``<cuda/cmath>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
   int ilog2(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
   int ceil_ilog2(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
   int ilog10(T value) noexcept;

   template <typename T>
   [[nodiscard]] __host__ __device__ __tile__ constexpr
   int ceil_ilog10(T value) noexcept;

   } // namespace cuda

The functions compute the logarithm to the base 2 and 10 of an integer value.

**Parameters**

- ``value``: The input value.

**Return value**

- ``ilog2``, ``ceil_ilog2``: The logarithm to the base 2, rounded down and up to the nearest integer respectively.
- ``ilog10``, ``ceil_ilog10``: The logarithm to the base 10, rounded down and up to the nearest integer respectively.

**Constraints**

- ``T`` is an integer type.

**Preconditions**

- ``value > 0``

**Performance considerations**

The functions perform the following operations in device code:

- ``ilog2``: ``FLO``
- ``ceil_ilog2``: ``FLO``, ``POPC``, ``ADD``, comparison
- ``ilog10``: ``FLO``, ``FMUL``, ``F2I``, constant memory lookup, ``SEL`` + ``IADD`` only if ``T == uint32_t`` or ``T == __uint128_t``
- ``ceil_ilog10``: ``ilog10`` with an additional comparison, subtraction, and ``IADD``

Example
-------

.. code:: cuda

    #include <cuda/cmath>
    #include <cuda/std/cassert>

    __global__ void ilog_kernel() {
        assert(cuda::ilog2(20) == 4);
        assert(cuda::ceil_ilog2(20) == 5);
        assert(cuda::ilog2(32) == 5);
        assert(cuda::ceil_ilog2(32) == 5);
        assert(cuda::ilog10(100) == 2);
        assert(cuda::ilog10(2000) == 3);
        assert(cuda::ceil_ilog10(100) == 2);
        assert(cuda::ceil_ilog10(2000) == 4);
    }

    int main() {
        ilog_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/7W3WaGd3c>`__

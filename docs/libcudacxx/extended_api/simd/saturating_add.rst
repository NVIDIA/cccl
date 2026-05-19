.. _libcudacxx-extended-api-simd-saturating-add:

``cuda::simd::saturating_add``
==============================

Defined in the ``<cuda/simd>`` header.

.. code:: cuda

   namespace cuda::simd {

   template <class T, class Abi>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::simd::basic_vec<T, Abi> saturating_add(
     const cuda::std::simd::basic_vec<T, Abi>& lhs,
     const cuda::std::simd::basic_vec<T, Abi>& rhs) noexcept;

   } // namespace cuda::simd

The function ``cuda::simd::saturating_add`` performs element-wise saturating addition of two ``cuda::std::simd::basic_vec`` objects.

For each element ``i`` in the input vectors, the result is equivalent to:

.. code:: cuda

   cuda::std::saturating_add(lhs[i], rhs[i])

**Parameters**

- ``lhs``: The left-hand side input vector.
- ``rhs``: The right-hand side input vector.

**Return value**

Returns a ``cuda::std::simd::basic_vec<T, Abi>`` where each element contains the saturated sum of the corresponding elements in ``lhs`` and ``rhs``.

**Constraints**

- ``T`` must be an `integer type <https://eel.is/c++draft/basic.fundamental#1>`__.

**Performance considerations**

- Packed 8-bit and 16-bit integer vectors perform saturating addition using ``VIADD.S8x4``, ``VIADD.U8x4``, ``VIADD.S16x2``, ``VIADD.U16x2`` on ``SM120f``.

Example
-------

.. code:: cuda

    #include <cuda/simd>
    #include <cuda/std/array>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>

    namespace simd = cuda::std::simd;

    __global__ void kernel()
    {
        using vec_t = simd::basic_vec<uint8_t, simd::fixed_size<4>>;

        cuda::std::array<uint8_t, 4> lhs_values{250, 10, 20, 30};
        cuda::std::array<uint8_t, 4> rhs_values{10, 20, 30, 40};
        vec_t lhs(lhs_values);
        vec_t rhs(rhs_values);

        vec_t result = cuda::simd::saturating_add(lhs, rhs);

        assert(result[0] == 255);
        assert(result[1] == 30);
        assert(result[2] == 50);
        assert(result[3] == 70);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

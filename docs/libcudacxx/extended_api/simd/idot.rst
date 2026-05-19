.. _libcudacxx-extended-api-simd-idot:

``cuda::simd::idot``
====================

Defined in the ``<cuda/simd>`` header.

.. code:: cuda

   namespace cuda::simd {

   template <class T, class U, class Abi, class AccT>
   [[nodiscard]] __host__ __device__ constexpr
   AccT idot(
     const cuda::std::simd::basic_vec<T, Abi>& lhs,
     const cuda::std::simd::basic_vec<U, Abi>& rhs,
     AccT acc) noexcept;

   } // namespace cuda::simd

The function ``cuda::simd::idot`` computes the dot product of two integer ``cuda::std::simd::basic_vec`` objects and adds the result to an accumulator.

For each element ``i`` in the input vectors, the result is equivalent to:

.. code:: cuda

   acc += static_cast<AccT>(lhs[i]) * static_cast<AccT>(rhs[i])

**Parameters**

- ``lhs``: The left-hand side input vector.
- ``rhs``: The right-hand side input vector.
- ``acc``: The initial accumulator value.

**Return value**

Returns ``acc`` plus the integer dot product of ``lhs`` and ``rhs``.

**Constraints**

- ``T``, ``U``, and ``AccT`` must be integer types.

**Performance considerations**

- Packed 8-bit input vectors with 32-bit accumulators (same sign) use ``IDP4A`` on ``SM61`` and newer device targets.
- Packed 16-bit by 8-bit input vectors with 32-bit accumulators (same sign) use ``IDP2A`` on ``SM61`` and newer device targets.
- Other integer input and accumulator combinations use the scalar fallback.

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
        using vec_t = simd::basic_vec<int8_t, simd::fixed_size<4>>;

        cuda::std::array<int8_t, 4> lhs_values{1, 2, 3, 4};
        cuda::std::array<int8_t, 4> rhs_values{5, 6, 7, 8};
        vec_t lhs(lhs_values);
        vec_t rhs(rhs_values);

        int32_t result = cuda::simd::idot(lhs, rhs, int32_t{10});

        assert(result == 80);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

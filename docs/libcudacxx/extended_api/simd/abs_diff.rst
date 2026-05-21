.. _libcudacxx-extended-api-simd-abs-diff:

``cuda::simd::abs_diff``
========================

Defined in the ``<cuda/simd>`` header.

.. code:: cuda

   namespace cuda::simd {

   template <class T, class Abi>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::simd::basic_vec<cuda::std::make_unsigned_t<T>, Abi> abs_diff(
     const cuda::std::simd::basic_vec<T, Abi>& lhs,
     const cuda::std::simd::basic_vec<T, Abi>& rhs) noexcept;

   } // namespace cuda::simd

The function ``cuda::simd::abs_diff`` performs element-wise absolute difference of two integer ``cuda::std::simd::basic_vec`` objects.

For each element ``i`` in the input vectors, the result is equivalent to:

.. code:: cuda

   abs(lhs[i] - rhs[i])

The return type is always an *unsigned* ``basic_vec`` with the same ABI as the input vectors.

**Parameters**

- ``lhs``: The left-hand side input vector.
- ``rhs``: The right-hand side input vector.

**Return value**

Returns a ``cuda::std::simd::basic_vec<cuda::std::make_unsigned_t<T>, Abi>`` where each element contains the unsigned absolute difference of the corresponding elements in ``lhs`` and ``rhs``.

**Constraints**

- ``T`` must be an integer type.

**Performance considerations**

- Packed 8-bit integer vectors perform absolute difference using:

  - ``VABSDIFF4`` on ``SM80``, ``SM86``, ``SM87``, ``SM89``, ``SM90``, ``SM100``, ``SM103``, and ``SM110``.
  - ``VIMNMX.S8x4/U8x4`` on ``SM120f``.

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
        using vec_t        = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<4>>;
        using result_vec_t = simd::basic_vec<cuda::std::uint8_t, simd::fixed_size<4>>;

        cuda::std::array<cuda::std::int8_t, 4> lhs_values{-128, 10, 20, 30};
        cuda::std::array<cuda::std::int8_t, 4> rhs_values{127, 20, -30, 40};

        vec_t lhs(lhs_values);
        vec_t rhs(rhs_values);
        result_vec_t result = cuda::simd::abs_diff(lhs, rhs);

        assert(result[0] == 255);
        assert(result[1] == 10);
        assert(result[2] == 50);
        assert(result[3] == 10);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

.. _libcudacxx-extended-api-simd-dot:

``cuda::simd::dot``
===================

Defined in the ``<cuda/simd>`` header.

.. code:: cuda

    namespace cuda::simd {

    template <class T, class U, class Abi, class AccT>
    [[nodiscard]] __host__ __device__ constexpr
    AccT dot(const cuda::std::simd::basic_vec<T, Abi>& lhs,
             const cuda::std::simd::basic_vec<U, Abi>& rhs,
             AccT                                      init) noexcept;

    } // namespace cuda::simd

The function ``cuda::simd::dot`` computes the dot product of two ``cuda::std::simd::basic_vec`` objects and adds the
result to an accumulator.

The result is equivalent to:

.. code:: cuda

    AccT acc = init;
    for (size_t i = 0; i < lhs.size(); ++i) {
        AccT lhs_value = static_cast<AccT>(lhs[i]);
        AccT rhs_value = static_cast<AccT>(rhs[i]);
        AccT product   = static_cast<AccT>(lhs_value * rhs_value);
        acc            = acc + product;
    }
    return acc;

**Parameters**

- ``lhs``: The left-hand side input vector.
- ``rhs``: The right-hand side input vector.
- ``init``: The initial accumulator value.

**Return value**

Returns ``init`` plus the dot product of ``lhs`` and ``rhs``.

**Constraints**

- The conversions from ``T`` and ``U`` to ``AccT``, multiplication of two ``AccT`` values, and addition of two ``AccT`` values must be well-formed.

**Undefined behavior**

- Arithmetic follows the rules of the operations on ``AccT``. In particular, signed integer overflow is undefined and
  unsigned integer results are reduced modulo the accumulator type's range.

**Performance considerations**

- Packed 8-bit integer input vectors with compatible 32-bit integer accumulators use ``IDP4A`` on ``SM61`` and newer
  device targets.
- Packed 16-bit by 8-bit integer input vectors with compatible 32-bit integer accumulators use ``IDP2A`` on ``SM61``
  and newer device targets.
- A compatible integer accumulator is unsigned when both inputs are unsigned and signed when either input is signed.
- Other input and accumulator combinations rely on vectorized multiplication (e.g. ``FMUL2`` with 32-bit floating-point types on ``SM100``) and addition instructions.

Example
-------

.. code:: cuda

    #include <cuda/simd>
    #include <cuda/std/array>
    #include <cuda/std/cassert>

    namespace simd = cuda::std::simd;

    __global__ void kernel()
    {
        using vec_t = simd::basic_vec<float, simd::fixed_size<4>>;

        cuda::std::array<float, 4> lhs_values{1.0f, 2.0f, 3.0f, 4.0f};
        cuda::std::array<float, 4> rhs_values{5.0f, 6.0f, 7.0f, 8.0f};
        vec_t lhs(lhs_values);
        vec_t rhs(rhs_values);

        float result = cuda::simd::dot(lhs, rhs, 10.0f);

        assert(result == 80.0f);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

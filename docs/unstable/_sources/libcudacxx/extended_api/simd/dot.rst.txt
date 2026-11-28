.. _libcudacxx-extended-api-simd-dot:

``cuda::simd::dot``
===================

Defined in the ``<cuda/simd>`` header.

.. code:: cuda

    namespace cuda::simd {

    template <class T, class U, class Abi, class AccT = cuda::std::common_type_t<T, U>>
    [[nodiscard]] __host__ __device__ constexpr
    AccT dot(const cuda::std::simd::basic_vec<T, Abi>& lhs,
             const cuda::std::simd::basic_vec<U, Abi>& rhs,
             AccT                                      init = {}) noexcept;

    } // namespace cuda::simd

The function ``cuda::simd::dot`` computes the dot product of two ``cuda::std::simd::basic_vec`` objects and adds the
result to an accumulator. If ``init`` is omitted, it is value-initialized and ``AccT`` defaults to
``cuda::std::common_type_t<T, U>``.

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
- ``init``: The initial accumulator value. Defaults to a value-initialized ``AccT``.

**Return value**

Returns ``init`` plus the dot product of ``lhs`` and ``rhs``.

**Constraints**

- The conversions from ``T`` and ``U`` to ``AccT``, multiplication of two ``AccT`` values, and addition of two ``AccT`` values must be well-formed.

**Undefined behavior**

- Arithmetic follows the rules of the operations on ``AccT``. In particular, signed integer overflow is undefined and
  unsigned integer results are reduced modulo the accumulator type's range.

**Performance considerations**

- Packed 8-bit integer input vectors with compatible 32-bit integer accumulators use ``IDP4A`` on  all device targets.
- Packed 16-bit by 8-bit integer input vectors with compatible 32-bit integer accumulators use ``IDP2A`` on all device targets.
- A compatible integer accumulator is unsigned when both inputs are unsigned and signed when either input is signed.
- Other input and accumulator combinations rely on vectorized multiplication (e.g. ``FMUL2`` with 32-bit floating-point types on ``SM100``) and addition instructions.

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
        using lhs_vec_t = simd::basic_vec<uint16_t, simd::fixed_size<4>>;
        using rhs_vec_t = simd::basic_vec<int8_t,   simd::fixed_size<4>>;

        cuda::std::array<uint16_t, 4> lhs_values{100, 200, 300, 400};
        cuda::std::array<int8_t, 4>   rhs_values{-1, 2, -3, 4};
        lhs_vec_t lhs(lhs_values);
        rhs_vec_t rhs(rhs_values);

        int result             = cuda::simd::dot(lhs, rhs);
        int accumulated_result = cuda::simd::dot(lhs, rhs, 10);

        assert(result == 1000);
        assert(accumulated_result == 1010);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

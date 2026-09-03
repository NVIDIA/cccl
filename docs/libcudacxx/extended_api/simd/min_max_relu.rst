.. _libcudacxx-extended-api-simd-min-max-relu:

``cuda::simd::min_relu`` and ``cuda::simd::max_relu``
=====================================================

Defined in the ``<cuda/simd>`` header.

.. code:: cuda

    namespace cuda::simd {

    template <class T, class Abi>
    [[nodiscard]] __host__ __device__ constexpr
    cuda::std::simd::basic_vec<T, Abi> max_relu(
        const cuda::std::simd::basic_vec<T, Abi>& lhs,
        const cuda::std::simd::basic_vec<T, Abi>& rhs) noexcept;

    template <class T, class Abi>
    [[nodiscard]] __host__ __device__ constexpr
    cuda::std::simd::basic_vec<T, Abi> min_relu(
        const cuda::std::simd::basic_vec<T, Abi>& lhs,
        const cuda::std::simd::basic_vec<T, Abi>& rhs) noexcept;

    template <class T, class Abi>
    [[nodiscard]] __host__ __device__ constexpr
    cuda::std::simd::basic_vec<T, Abi> max_relu(
        const cuda::std::simd::basic_vec<T, Abi>& a,
        const cuda::std::simd::basic_vec<T, Abi>& b,
        const cuda::std::simd::basic_vec<T, Abi>& c) noexcept;

    template <class T, class Abi>
    [[nodiscard]] __host__ __device__ constexpr
    cuda::std::simd::basic_vec<T, Abi> min_relu(
        const cuda::std::simd::basic_vec<T, Abi>& a,
        const cuda::std::simd::basic_vec<T, Abi>& b,
        const cuda::std::simd::basic_vec<T, Abi>& c) noexcept;

   } // namespace cuda::simd

The functions perform an element-wise minimum or maximum followed by ReLU.
For each element ``i``, the two-input overloads are equivalent to:

.. code:: cuda

   max_relu(lhs, rhs)[i] == cuda::std::max(cuda::std::max(lhs[i], rhs[i]), T{0})
   min_relu(lhs, rhs)[i] == cuda::std::max(cuda::std::min(lhs[i], rhs[i]), T{0})

The three-input overloads are equivalent to:

.. code:: cuda

   max_relu(a, b, c)[i] == cuda::std::max(cuda::std::max(cuda::std::max(a[i], b[i]), c[i]), T{0})
   min_relu(a, b, c)[i] == cuda::std::max(cuda::std::min(cuda::std::min(a[i], b[i]), c[i]), T{0})

The functionalities map to [Dynamic Programming eXtension (DPX)](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#dynamic-programming-extension-dpx-instructions) instructions.

**Parameters**

- ``lhs``, ``rhs``: The input vectors for the two-input overloads.
- ``a``, ``b``, ``c``: The input vectors for the three-input overloads.

**Return value**

Returns a ``cuda::std::simd::basic_vec<T, Abi>`` containing the element-wise result.

**Constraints**

- ``T`` must be a signed integer type.

**Performance considerations**

The following cases are optimized on the device:

- Signed 8-bit elements on ``SM107f`` and ``SM120f``:

  - Two-input overload uses one ``VIMNMX.S8x4.RELU`` instruction per four elements.
  - Three-input overload uses two ``VIMNMX.S8x4.RELU`` instructions per four elements.

- Signed 16-bit elements:

  - On ``SM90``, ``SM100``, ``SM103``, ``SM107``, and ``SM120``, each two-input overload uses one ``VIMNMX.S16x2.RELU`` instruction per two elements.
  - On ``SM90``, ``SM100``, and ``SM103``, each three-input overload uses one ``VIMNMX3.S16x2.RELU`` instruction per two elements.
  - On ``SM107`` and ``SM120``, each three-input overload uses two ``VIMNMX.S16x2.RELU`` instructions per two elements.

- Signed 32-bit elements:

  - On ``SM90``, ``SM100``, ``SM103``, ``SM107``, and ``SM120``, each two-input overload uses one ``VIMNMX.RELU`` instruction per element.
  - On ``SM90``, ``SM100``, and ``SM103``, each three-input overload uses one ``VIMNMX3.RELU`` instruction per element.
  - On ``SM107`` and ``SM120``, each three-input overload uses two ``VIMNMX.RELU`` instructions per element.

Example
-------

.. code:: cuda

    #include <cuda/simd>
    #include <cuda/std/array>
    #include <cuda/std/cassert>
    #include <cuda/std/cstdint>

    #include <cuda_runtime_api.h>

    namespace simd = cuda::std::simd;

    __global__ void kernel()
    {
        using vec_t = simd::basic_vec<int16_t, simd::fixed_size<2>>;

        vec_t a(cuda::std::array<int16_t, 2>{-4, 8});
        vec_t b(cuda::std::array<int16_t, 2>{-2, 3});
        vec_t c(cuda::std::array<int16_t, 2>{-6, 5});

        vec_t maximum = cuda::simd::max_relu(a, b, c);
        vec_t minimum = cuda::simd::min_relu(a, b, c);

        assert(maximum[0] == 0);
        assert(maximum[1] == 8);
        assert(minimum[0] == 0);
        assert(minimum[1] == 3);
    }

    int main()
    {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

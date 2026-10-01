.. _libcudacxx-extended-api-simd-add-min-max:

``cuda::simd::add_min``, ``cuda::simd::add_max``, ``cuda::simd::add_min_relu``, and ``cuda::simd::add_max_relu``
================================================================================================================

Defined in the ``<cuda/simd>`` header.

.. code:: cuda

   namespace cuda::simd {

   template <class T, class Abi>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::simd::basic_vec<T, Abi> add_max(
     const cuda::std::simd::basic_vec<T, Abi>& a,
     const cuda::std::simd::basic_vec<T, Abi>& b,
     const cuda::std::simd::basic_vec<T, Abi>& c) noexcept;

   template <class T, class Abi>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::simd::basic_vec<T, Abi> add_min(
     const cuda::std::simd::basic_vec<T, Abi>& a,
     const cuda::std::simd::basic_vec<T, Abi>& b,
     const cuda::std::simd::basic_vec<T, Abi>& c) noexcept;

   template <class T, class Abi>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::simd::basic_vec<T, Abi> add_max_relu(
     const cuda::std::simd::basic_vec<T, Abi>& a,
     const cuda::std::simd::basic_vec<T, Abi>& b,
     const cuda::std::simd::basic_vec<T, Abi>& c) noexcept;

   template <class T, class Abi>
   [[nodiscard]] __host__ __device__ constexpr
   cuda::std::simd::basic_vec<T, Abi> add_min_relu(
     const cuda::std::simd::basic_vec<T, Abi>& a,
     const cuda::std::simd::basic_vec<T, Abi>& b,
     const cuda::std::simd::basic_vec<T, Abi>& c) noexcept;

   } // namespace cuda::simd

The functions perform an element-wise addition followed by a minimum or maximum, optionally followed by ReLU.
For each element ``i``, the functions are equivalent to:

.. code:: cuda

   add_max(a, b, c)[i]      == cuda::std::max(a[i] + b[i], c[i])
   add_min(a, b, c)[i]      == cuda::std::min(a[i] + b[i], c[i])
   add_max_relu(a, b, c)[i] == cuda::std::max(cuda::std::max(a[i] + b[i], c[i]), T{0})
   add_min_relu(a, b, c)[i] == cuda::std::max(cuda::std::min(a[i] + b[i], c[i]), T{0})

On supported GPU architectures, the optimized device paths map to `Dynamic Programming eXtension (DPX) <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#dynamic-programming-extension-dpx-instructions>`__ instructions.

**Parameters**

- ``a``, ``b``: The vectors whose corresponding elements are added.
- ``c``: The vector compared with the element-wise sum.

**Return value**

Returns a ``cuda::std::simd::basic_vec<T, Abi>`` containing the element-wise result.

**Constraints**

- ``add_min`` and ``add_max``: ``T`` must be an `integer type <https://eel.is/c++draft/basic.fundamental#1>`__.
- ``add_min_relu`` and ``add_max_relu``: ``T`` must be a signed integer type.

**Performance considerations**

On ``SM90``, ``SM100``, and ``SM103``:

- Signed and unsigned 16-bit elements use one ``VIADDMNMX.S16x2`` or ``VIADDMNMX.U16x2`` instruction per two elements.
- Signed and unsigned 32-bit elements use one ``VIADDMNMX`` instruction per element.
- ``add_min_relu`` and ``add_max_relu`` use one ``VIADDMNMX.S16x2.RELU`` instruction per two signed 16-bit elements and one ``VIADDMNMX.RELU`` instruction per signed 32-bit element.

On ``SM107`` and ``SM120``:

- Signed and unsigned 16-bit elements use one ``VIADD.16x2`` and one ``VIMNMX.S16x2`` or ``VIMNMX.U16x2`` instruction per two elements.
- Signed and unsigned 32-bit elements use one ``IADD`` and one ``VIMNMX`` instruction per element.
- ``add_min_relu`` and ``add_max_relu`` use one addition and one ``VIMNMX.S16x2.RELU`` or ``VIMNMX.S32.RELU`` instruction per two signed 16-bit elements or per signed 32-bit element.

Other element types use the portable element-wise implementation.

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
     vec_t b(cuda::std::array<int16_t, 2>{1, -3});
     vec_t c(cuda::std::array<int16_t, 2>{-2, 10});

     vec_t maximum      = cuda::simd::add_max(a, b, c);
     vec_t minimum      = cuda::simd::add_min(a, b, c);
     vec_t maximum_relu = cuda::simd::add_max_relu(a, b, c);
     vec_t minimum_relu = cuda::simd::add_min_relu(a, b, c);

     assert(maximum[0] == -2);
     assert(maximum[1] == 10);
     assert(minimum[0] == -3);
     assert(minimum[1] == 5);
     assert(maximum_relu[0] == 0);
     assert(maximum_relu[1] == 10);
     assert(minimum_relu[0] == 0);
     assert(minimum_relu[1] == 5);
   }

   int main()
   {
     kernel<<<1, 1>>>();
     cudaDeviceSynchronize();
   }

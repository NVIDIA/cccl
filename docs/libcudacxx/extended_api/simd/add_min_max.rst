.. _libcudacxx-extended-api-simd-add-min-max:

``cuda::simd::add_min`` and ``cuda::simd::add_max``
===================================================

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

   } // namespace cuda::simd

The functions perform an element-wise addition followed by a minimum or maximum.
For each element ``i``, the functions are equivalent to:

.. code:: cuda

   add_max(a, b, c)[i] == cuda::std::max(a[i] + b[i], c[i])
   add_min(a, b, c)[i] == cuda::std::min(a[i] + b[i], c[i])

A ReLU form can be obtained by composing the addition with :ref:`cuda::simd::min_relu and cuda::simd::max_relu <libcudacxx-extended-api-simd-min-max-relu>`:

.. code:: cuda

   auto maximum_relu = cuda::simd::max_relu(a + b, c);
   auto minimum_relu = cuda::simd::min_relu(a + b, c);

On supported GPU architectures, the optimized device paths map to `Dynamic Programming eXtension (DPX) <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#dynamic-programming-extension-dpx-instructions>`__ instructions.

**Parameters**

- ``a``, ``b``: The vectors whose corresponding elements are added.
- ``c``: The vector compared with the element-wise sum.

**Return value**

Returns a ``cuda::std::simd::basic_vec<T, Abi>`` containing the element-wise result.

**Constraints**

- ``T`` must be an integer type.
- The composed ReLU forms require ``T`` to be a signed integer type.

**Performance considerations**

On ``SM90``, ``SM100``, and ``SM103``:

- Signed and unsigned 16-bit elements use one ``VIADDMNMX.S16x2`` or ``VIADDMNMX.U16x2`` instruction per two elements.
- Signed and unsigned 32-bit elements use one ``VIADDMNMX`` instruction per element.
- The composed signed 16-bit and 32-bit ReLU ``cuda::simd::max_relu(a + b, c)`` and ``cuda::simd::min_relu(a + b, c)`` forms use the corresponding ``VIADDMNMX.RELU`` instruction.

On ``SM107`` and ``SM120``:

- Signed and unsigned 16-bit elements use one ``VIADD.16x2`` and one ``VIMNMX.S16x2`` or ``VIMNMX.U16x2`` instruction.
- Signed and unsigned 32-bit elements use one ``IADD`` and one ``VIMNMX`` instruction.
- The composed signed 16-bit and 32-bit ReLU ``cuda::simd::max_relu(a + b, c)`` and ``cuda::simd::min_relu(a + b, c)`` forms use one addition and one of the corresponding ``VIMNMX.RELU`` instruction.

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
     vec_t maximum_relu = cuda::simd::max_relu(a + b, c);
     vec_t minimum_relu = cuda::simd::min_relu(a + b, c);

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

.. _libcudacxx-extended-api-math-ceil-div:

Math
=====

.. code:: cuda

   template <typename T, typename U>
   [[nodiscard]] __host__ __device__ inline
   constexpr _CUDA_VSTD::common_type_t<_Tp, _Up> ceil_div(T a, U b) noexcept;

ceil_div
---------

- *Requires*: ``T`` is an integral type (including 128-bit integers) or enumerator.
- *Preconditions*: ``a >= 0`` is true and ``b > 0`` is true.
- *Returns*: divides ``a`` by ``b``. If ``a`` is not a multiple of ``b`` rounds the result up to the next integer value.

**Performance considerations**

- The function computes ``(a + b - 1) / b`` when the common type is a signed integer.
- The function computes ``min(a, 1 + ((a - 1) / b)`` when the common type is an unsigned integer in CUDA, which generates less instructions than ``(a / b) + ((a / b) * b != a)``, especially for 64-bit types.

**Example**: This API is very useful for determining the *number of thread blocks* required to process a fixed amount of work, given a fixed number of threads per block:

.. code:: cuda

   #include <vector>
   #include <cuda/cmath>

   __global__ void vscale(int n, float s, float *x) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) x[i] *= s;
   }

   int main() {
     const int n = 100000;
     const float s = 2.f;
     std::vector<float> x(n, 1.f);

     // Given a fixed number of threads per block...
     constexpr int threads_per_block = 256;

     // ...dividing some "n" by "threads_per_block" may lead to a remainder,
     // requiring the kernel to be launched with an extra thread block to handle it.
     const int thread_blocks = cuda::ceil_div(n, threads_per_block);

     vscale<<<thread_blocks, threads_per_block>>>(n, s, x.data());
     cudaDeviceSynchronize();

     return 0;
   }

`See it on Godbolt TODO`

.. _libcudacxx-extended-api-math:

Math
=====

.. code:: cuda

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr T ceil_div(T a, T b) noexcept;

ceil_div
---------

- _Requires_: `is_integral_v<T>` is true.
- _Preconditions_: `a >= 0` is true and `b > 0` is true.
- _Returns_: divides `a` by `b`. If `a` is not a multiple of `b` rounds the result up to the next integer value.

.. note::

   The function is only constexpr from C++14 onwards

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

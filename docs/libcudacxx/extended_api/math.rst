.. _libcudacxx-extended-api-math:

Math
=============

.. code:: cuda

   template <typename T>
   __host__ __device__ constexpr T ceil_div(T a, T b) noexcept(true);

`ceil_div`
------------

- _Mandates_: `is_integral_v<T>` is true.
- _Preconditions_: `a >= 0` is true and `b > 0` is true.
- _Returns_: divides `a` by `b` and rounds the result up to a multiple of `a`.

**Example**: This API is very useful for determining the _number of thread blocks_ required to process a fixed amount of work, given a fixed number of threads per block:

.. code:: cuda

   #include <vector>
   #include <cuda/cmath>

   __global__ void vscale(int n, float s, float *x) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) x[i] *= s;
   }

   int main() {
     int n = 100000;
     float s = 2.f;
     std::vector<float> x(n, 1.f);

     // Given a fixed number of threads per block...
     int threads_per_block = 256;

     // ...dividing some "n" by "threads_per_block" may lead to a remainder,
     // requiring the kernel to be launched with an extra thread block to handle it.
     int thread_blocks = cuda::ceil_div(n, threads_per_block);

     vscale<<<thread_blocks, threads_per_block>>>(n, s, x.data());
     cudaDeviceSynchronize();

     return 0;
   }

`See it on Godbolt TODO`

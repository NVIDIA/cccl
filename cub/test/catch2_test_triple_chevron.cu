// #include <thrust/count.h>
// #include <cub/detail/triple_chevron_launch.cuh>
// #include <cuda/std/tuple>

// // Has to go after all cub headers. Otherwise, this test won't catch unused
// // variables in cub kernels.
// #include "catch2_test_cdp_helper.h"
// #include "catch2_test_helper.h"

// // %PARAM% TEST_CDP cdp 0:1

// struct Functor {
//   cub::detail::triple_chevron chev;

//   template <class T, class Kernel, class... Args>
//   cudaError_t operator()(T d_in, T d_out, Kernel k, int n, Args ... args) const {
//       return chev.doit(k, d_in, d_out, n, args ...); 
//   }

//   template <class T, class Kernel, class... Args>
//   cudaError_t operator()(uint8_t*, size_t, Kernel k, T d_in, T d_out, int n, Args ... args) const {
//       return chev.doit(k, d_in, d_out, n, args ...); 
//   }
// };

// template <class T>
// __global__ void api_kernel(const T *d_in, T *d_out, int num_items)
// {
//   const int i = blockIdx.x * blockDim.x + threadIdx.x;

//   if (i < num_items)
//   {
//     d_out[i] = d_in[i] * T{2};
//   }
// }

// CUB_TEST("Triple Chevron launch kernels from host and device", "[test][utils]") {
//   int n = 42;
//   thrust::device_vector<int> in(n, 21);
//   thrust::device_vector<int> out(n);
//   int *d_in  = thrust::raw_pointer_cast(in.data());
//   int *d_out = thrust::raw_pointer_cast(out.data());

//   const int block_size = 256;
//   const int grid_size = (n * block_size - 1) / block_size;
//   auto chev = cub::detail::triple_chevron(grid_size, block_size, 0, 0);
//   Functor functor{chev};
//   constexpr bool on_device = TEST_CDP;
  
//   cdp_launch(functor, api_kernel<int>, d_in, d_out, n, on_device);

//   REQUIRE(1==2);
// }

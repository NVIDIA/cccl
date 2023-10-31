#include <thrust/count.h>
#include <cub/detail/triple_chevron_launch.cuh>
#include <cuda/std/tuple>
#include <cstdio>  

// Has to go after all cub headers. Otherwise, this test won't catch unused
// variables in cub kernels.
#include "catch2_test_cdp_helper.h"
#include "catch2_test_helper.h"

// %PARAM% TEST_CDP cdp 0:1

__global__ void add_kernel(int a, float b, double* out) {
    *out = a + b;
}

__global__ void shared_mem_kernel(unsigned int* size) {
  extern __shared__ char smem[];

  unsigned int smem_size{};
  asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(smem_size));

  *size = smem_size;
}

template <class T>
__global__ void mult_two_kernel(const T *d_in, T *d_out, int num_items)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_items)
    {
        d_out[i] = d_in[i] * T{2};
    }
}


struct cdp_chevron_invoker { 
  static constexpr int threads_in_block = 256;

  template <class T, class KernelT>
  CUB_RUNTIME_FUNCTION static cudaError_t invoke(std::uint8_t *d_temp_storage,
                                                 std::size_t &temp_storage_bytes,
                                                 KernelT kernel,
                                                 const T *d_in,
                                                 T *d_out,
                                                 int num_items,
                                                 bool on_device)
  {
    NV_IF_TARGET(NV_IS_HOST,
                 (if (on_device) { return cudaErrorLaunchFailure; }),
                 (if (!on_device) { return cudaErrorLaunchFailure; }));

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = static_cast<std::size_t>(num_items);
      return cudaSuccess;
    }
  
    if (temp_storage_bytes != static_cast<std::size_t>(num_items))
    {
      return cudaErrorInvalidValue;
    }

    const int blocks_in_grid = (num_items + threads_in_block - 1) / threads_in_block;

    return cub::detail::triple_chevron(blocks_in_grid, threads_in_block, 0, 0)
      .doit(kernel, d_in, d_out, num_items);
  }

  template <class T>
  CUB_RUNTIME_FUNCTION static cudaError_t create(std::uint8_t *d_temp_storage,
                                               std::size_t &temp_storage_bytes,
                                               const T *d_in,
                                               T *d_out,
                                               int num_items,
                                               bool device_invoke)
  {
    return invoke(d_temp_storage,
                  temp_storage_bytes,
                  mult_two_kernel<T>,
                  d_in,
                  d_out,
                  num_items,
                  device_invoke);
  }
};

struct cdp_invocable {
    static constexpr int threads_in_block = 256;

    template <class T, class KernelT>
    CUB_RUNTIME_FUNCTION cudaError_t operator()(
        uint8_t* d_temp_storage,
        size_t& temp_storage_bytes,
        KernelT kernel,
        const T* d_in,
        T* d_out,
        int num_items,
        bool on_device) const 
    {
        NV_IF_TARGET(NV_IS_HOST,
                    (if (on_device) { return cudaErrorLaunchFailure; }),
                    (if (!on_device) { return cudaErrorLaunchFailure; }));

        if (d_temp_storage == nullptr) 
        {
            temp_storage_bytes = static_cast<std::size_t>(num_items);
            return cudaSuccess;
        }

        if (temp_storage_bytes != static_cast<std::size_t>(num_items))
        {
            return cudaErrorInvalidValue;
        }

        const int blocks_in_grid = (num_items + threads_in_block - 1) / threads_in_block;

        return cub::detail::triple_chevron(blocks_in_grid, threads_in_block, 0, 0)
            .doit(kernel, d_in, d_out, num_items);
    }
};


CUB_TEST("CDP wrapper works with custom invocables and cdp_launch, on both host and device", "[device][triple_chevron]")
{
  int n = 42;
  thrust::device_vector<int> in(n, 21);
  thrust::device_vector<int> out(n);

  int *d_in  = thrust::raw_pointer_cast(in.data());
  int *d_out = thrust::raw_pointer_cast(out.data());

  constexpr bool on_device = TEST_CDP;

  {
    cdp_launch(cdp_invocable{}, mult_two_kernel<int>, d_in, d_out, n, on_device);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    std::vector<int> h_out(n);
    thrust::copy(out.begin(), out.end(), h_out.begin());

    const auto actual   = static_cast<std::size_t>(thrust::count(out.begin(), out.end(), 42));
    const auto expected = static_cast<std::size_t>(n);

    REQUIRE(actual == expected);
  }


}


CUB_TEST("Triple Chevron properly launches with cuda streams involved and returns successful", "[device][triple_chevron]") {
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  
  cudaEvent_t event;
  cudaEventCreate(&event);

  int n = 1000;
  thrust::device_vector<int> in(n, 5);
  thrust::device_vector<int> out(n);

  int *d_in  = thrust::raw_pointer_cast(in.data());
  int *d_out = thrust::raw_pointer_cast(out.data());

  cudaMemcpyAsync(d_out, d_in, n * sizeof(float), cudaMemcpyHostToDevice, stream1); 
  cudaEventRecord(event, stream1); 

  const int block_size = 256;
  const int grid_size = (n * block_size - 1) / block_size;
  auto chev = cub::detail::triple_chevron(grid_size, block_size, 0, stream2);

  cudaStreamWaitEvent(stream2, event, 0); // sync streams
  chev.doit(mult_two_kernel<int>, d_in, d_out, n);

  {
    const auto actual   = static_cast<std::size_t>(thrust::count(out.begin(), out.end(), 10));
    const auto expected   = static_cast<std::size_t>(n);
    REQUIRE(actual==expected);
  }
  cudaEventDestroy(event);

}

CUB_TEST("Triple chevron returns kernel launch failures", "[test][utils]") {
  int n = 42;
  thrust::device_vector<int> in(n, 21);
  thrust::device_vector<int> out(n);
  int *d_in  = thrust::raw_pointer_cast(in.data());
  int *d_out = thrust::raw_pointer_cast(out.data());

  auto chevron = cub::detail::triple_chevron(0, 0);
  auto err = chevron.doit(mult_two_kernel<int>, d_in, d_out, n);
  REQUIRE( CubDebug(cudaErrorMissingConfiguration) == cudaErrorMissingConfiguration );
}

CUB_TEST("Triple Chevron respects required dynamic shared memory allocation", "[test][utils]") {

  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  auto cap = deviceProperties.sharedMemPerBlock;

  auto chevron = cub::detail::triple_chevron(1, 1, cap); 
  unsigned int host_actual;
  unsigned int* device_actual;

  cudaMalloc(&device_actual, sizeof(unsigned int)); 
  auto err = chevron.doit(shared_mem_kernel, device_actual);
  cudaMemcpy(&host_actual, device_actual, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  REQUIRE(host_actual == cap);

  cudaFree(device_actual);
}

CUB_TEST("Triple Chevron properly forwards parameters", "[device][triple_chevron]") {
  double result;
  double *d_result;

  cudaMalloc(&d_result, sizeof(double));
  auto chev = cub::detail::triple_chevron(1, 1); 
  chev.doit(add_kernel, 5, 3.5f, d_result);
  cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(result == 8.5f);
}

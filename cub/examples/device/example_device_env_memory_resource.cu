// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Example of DeviceReduce::Sum() using an environment with an explicitly provided
/// memory resource, including a synchronous cudaMalloc-based fallback for devices
/// without cudaMallocAsync support (e.g. Windows GPUs in TCC driver mode).

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstdio>
#include <stdexcept>

#include "../../test/test_util.h"

// example-begin env-mr-fallback-definition
// A synchronous memory resource on top of cudaMalloc/cudaFree. Every allocation and
// deallocation blocks the calling thread and synchronizes the device; passing it
// explicitly makes that behavior a visible, deliberate choice. Allocations are made
// on the device that is current at the time of the call.
struct synchronous_memory_resource : cuda::mr::memory_resource_base<synchronous_memory_resource>
{
  [[nodiscard]] void* allocate_sync(size_t bytes, size_t alignment = cuda::mr::default_cuda_malloc_alignment)
  {
    if (alignment > cuda::mr::default_cuda_malloc_alignment || cuda::mr::default_cuda_malloc_alignment % alignment != 0)
    {
      throw std::invalid_argument("invalid alignment for synchronous_memory_resource");
    }

    void* ptr          = nullptr;
    cudaError_t status = cudaMalloc(&ptr, bytes);
    if (status != cudaSuccess)
    {
      throw std::runtime_error(cudaGetErrorString(status));
    }

    return ptr;
  }

  void deallocate_sync(void* ptr,
                       [[maybe_unused]] size_t bytes,
                       [[maybe_unused]] size_t alignment = cuda::mr::default_cuda_malloc_alignment) noexcept
  {
    cudaFree(ptr);
  }

  friend constexpr void get_property(synchronous_memory_resource const&, cuda::mr::device_accessible) noexcept {}

  friend constexpr bool operator==(synchronous_memory_resource, synchronous_memory_resource) noexcept
  {
    return true;
  }

#if _CCCL_STD_VER <= 2017
  friend constexpr bool operator!=(synchronous_memory_resource, synchronous_memory_resource) noexcept
  {
    return false;
  }
#endif
};

// Selects the default memory pool when the device supports cudaMallocAsync, and the
// synchronous fallback otherwise.
cuda::mr::any_resource<cuda::mr::device_accessible> make_device_resource(cuda::device_ref dev)
{
  if (dev.attribute(cuda::device_attributes::memory_pools_supported))
  {
    return cuda::mr::any_resource<cuda::mr::device_accessible>{cuda::device_default_memory_pool(dev)};
  }

  return cuda::mr::make_any_resource<cuda::mr::synchronous_resource_adapter<synchronous_memory_resource>,
                                     cuda::mr::device_accessible>(synchronous_memory_resource{});
}
// example-end env-mr-fallback-definition

bool g_verbose = false; // Whether to display input/output to console

int main(int argc, char** argv)
{
  // Initialize command line and print usage
  CommandLineArgs args(argc, argv);
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--n=<input items> "
           "[--v] "
           "\n",
           argv[0]);
    std::exit(0);
  }

  // Parse command line options
  int num_items = 150;
  g_verbose     = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("n", num_items);

  printf("cub::DeviceReduce::Sum() %d items (%d-byte elements)\n", num_items, (int) sizeof(int));
  fflush(stdout);

  // Initialize problem and solution
  std::vector<int> h_in(num_items);
  int h_reference = 0;
  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = i;
    h_reference += i;
  }

  // Allocate and initialize device arrays
  auto d_in = thrust::device_vector<int>(num_items, thrust::no_init);
  thrust::copy(h_in.begin(), h_in.end(), d_in.begin());
  auto d_out = thrust::device_vector<int>(1);

  auto device = cuda::devices[0];
  auto stream = cuda::stream{device};

  // example-begin env-mr-fallback-run
  auto mr  = make_device_resource(device);
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, mr};

  const cudaError_t error = cub::DeviceReduce::Sum(d_in.data(), d_out.data(), num_items, env);
  // example-end env-mr-fallback-run
  CubDebugExit(error);

  // Check for correctness (and display results, if specified)
  const int compare =
    CompareDeviceResults(&h_reference, thrust::raw_pointer_cast(d_out.data()), 1, g_verbose, g_verbose);
  printf("\t%s", compare ? "FAIL" : "PASS");
  printf("\n\n");

  return compare;
}

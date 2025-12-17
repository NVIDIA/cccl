// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Simple example of cub::DeviceTopK::MinKeys().
//! Find the top-k smallest float keys paired with a corresponding array of int values.
//! To compile using the command line:
//!   nvcc -arch=sm_XX example_device_topk_keys.cu -I../.. -lcudart -O3

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_topk.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cuda/stream>

#include <algorithm>
#include <iostream>

#include "../../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------

// Whether to display input/output to console
bool g_verbose = false;

//---------------------------------------------------------------------
// Helper functions
//---------------------------------------------------------------------

// Initialize the input data and the reference solution
void initialize(float* h_keys, float* h_reference_keys, int num_items, int k)
{
  for (int i = 0; i < num_items; ++i)
  {
    RandomBits(h_keys[i]);
  }

  if (g_verbose)
  {
    std::cout << "Input keys:\n";
    DisplayResults(h_keys, num_items);
    std::cout << "\n\n";
  }

  std::partial_sort_copy(h_keys, h_keys + num_items, h_reference_keys, h_reference_keys + k);
}

//  In this example, we do no require a specific output order and do not require deterministic results (this allows for
//  better performance in some cases). However, the output of DeviceTopK::MinKeys() is not sorted. This function sorts
//  the output keys for comparison against the reference solution.
thrust::host_vector<float> sort_unordered_results(thrust::host_vector<float> h_res_keys)
{
  thrust::sort(h_res_keys.begin(), h_res_keys.end());
  return h_res_keys;
}

//---------------------------------------------------------------------
// Main
//--------------------------------------------------------------------
int main(int argc, char** argv)
{
  int num_items = 10240;
  int k         = 10;

  // initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("k", k);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    std::cout << "Usage: " << argv[0] << " [--n=<input items>] [--k=<output items>] [--device=<device-id>] [--v]\n";
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  std::cout << "cub::DeviceTopK::MinKeys() find " << k << " smallest items from " << num_items << " items ("
            << sizeof(float) << "-byte keys)\n";

  // Allocate host arrays
  thrust::host_vector<float> h_keys_vector(num_items);
  thrust::host_vector<float> h_reference_keys_vector(k);

  // Initialize problem and solution on host
  initialize(thrust::raw_pointer_cast(h_keys_vector.data()),
             thrust::raw_pointer_cast(h_reference_keys_vector.data()),
             num_items,
             k);

  // Allocate device arrays
  thrust::device_vector<float> d_keys_in{h_keys_vector};
  thrust::device_vector<float> d_keys_out(k);

  // Specify that we do not require a specific output order and do not require deterministic results
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  // Prepare CUDA stream
  cudaStream_t stream = nullptr;
  CubDebugExit(cudaStreamCreate(&stream));
  cuda::stream_ref stream_ref{stream};

  // Create the environment with the stream and requirements
  auto env = cuda::std::execution::env{stream_ref, requirements};

  // Query temporary storage requirements
  size_t temp_storage_bytes = 0;
  CubDebugExit(
    DeviceTopK::MinKeys(nullptr, temp_storage_bytes, d_keys_in.begin(), d_keys_out.begin(), num_items, k, env));

  // Allocate temporary storage
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  void* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run the top-k algorithm
  CubDebugExit(
    DeviceTopK::MinKeys(d_temp_storage, temp_storage_bytes, d_keys_in.begin(), d_keys_out.begin(), num_items, k, env));

  // Check for correctness (and display results, if specified)
  auto h_res_keys_vector = sort_unordered_results(d_keys_out);
  if (g_verbose)
  {
    std::cout << "Output keys:\n";
    DisplayResults(thrust::raw_pointer_cast(h_res_keys_vector.data()), k);
    std::cout << "\n\n";
  }
  const int compare = CompareResults(h_reference_keys_vector.data(), h_res_keys_vector.data(), k, g_verbose);
  AssertEquals(0, compare);

  std::cout << "\n\n";

  return 0;
}

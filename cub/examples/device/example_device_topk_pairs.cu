// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Simple example of cub::DeviceTopK::MinPairs().
//! Find the top-k smallest float keys paired with a corresponding array of int values.
//! To compile using the command line:
//!   nvcc -arch=sm_XX example_device_topk_pairs.cu -I../.. -lcudart -O3

// Ensure printing of CUDA runtime errors to console

#define CUB_STDERR

#include <cub/device/device_topk.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <cuda/std/tuple>
#include <cuda/stream>

#include <algorithm>
#include <iostream>

#include "../../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

// Whether to display input/output to console
bool g_verbose = false;

//---------------------------------------------------------------------
// Helper functions
//---------------------------------------------------------------------

// Initialize key-value sorting problem.
void initialize(float* h_keys, int* h_values, float* h_reference_keys, int* h_reference_values, int num_items, int k)
{
  for (int i = 0; i < num_items; ++i)
  {
    RandomBits(h_keys[i]);
    RandomBits(h_values[i]);
  }

  if (g_verbose)
  {
    std::cout << "Input keys:\n";
    DisplayResults(h_keys, num_items);
    std::cout << "\n\n";

    std::cout << "Input values:\n";
    DisplayResults(h_values, num_items);
    std::cout << "\n\n";
  }

  auto h_pairs           = thrust::make_zip_iterator(h_keys, h_values);
  auto h_reference_pairs = thrust::make_zip_iterator(h_reference_keys, h_reference_values);
  std::partial_sort_copy(h_pairs, h_pairs + num_items, h_reference_pairs, h_reference_pairs + k);
}

//  In this example, we do no require a specific output order and do not require deterministic results (this allows for
//  better performance in some cases). However, the output of DeviceTopK::MinPairs() is not sorted. This function sorts
//  the output keys for comparison against the reference solution.
::cuda::std::tuple<thrust::host_vector<float>, thrust::host_vector<int>>
sort_unordered_results(thrust::host_vector<float> h_res_keys, thrust::host_vector<int> h_res_values)
{
  auto h_pairs = thrust::make_zip_iterator(h_res_keys.begin(), h_res_values.begin());
  thrust::sort(h_pairs, h_pairs + h_res_keys.size());
  return ::cuda::std::make_tuple(h_res_keys, h_res_values);
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv)
{
  int num_items = 10240;
  int k         = 10;

  // Initialize command line
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

  std::cout << "cub::DeviceTopK::MinPairs() find " << k << " smallest items from " << num_items << " items ("
            << sizeof(float) << "-byte keys " << sizeof(int) << "-byte values)\n";

  // Allocate host arrays
  thrust::host_vector<float> h_keys_vector(num_items);
  thrust::host_vector<float> h_reference_keys_vector(k);
  thrust::host_vector<float> h_res_keys_vector(k);
  thrust::host_vector<int> h_values_vector(num_items);
  thrust::host_vector<int> h_reference_values_vector(k);
  thrust::host_vector<int> h_res_values_vector(k);

  // Initialize problem and solution on host
  initialize(h_keys_vector.data(),
             h_values_vector.data(),
             h_reference_keys_vector.data(),
             h_reference_values_vector.data(),
             num_items,
             k);

  // Allocate device arrays
  thrust::device_vector<float> d_keys_in{h_keys_vector};
  thrust::device_vector<int> d_values_in{h_values_vector};
  thrust::device_vector<float> d_keys_out(k);
  thrust::device_vector<int> d_values_out(k);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;

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
  CubDebugExit(DeviceTopK::MinPairs(
    nullptr,
    temp_storage_bytes,
    d_keys_in.begin(),
    d_keys_out.begin(),
    d_values_in.begin(),
    d_values_out.begin(),
    num_items,
    k,
    env));

  // Allocate temporary storage
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  void* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run the top-k algorithm
  CubDebugExit(DeviceTopK::MinPairs(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in.begin(),
    d_keys_out.begin(),
    d_values_in.begin(),
    d_values_out.begin(),
    num_items,
    k,
    env));

  // Check for correctness (and display results, if specified)
  auto [h_res_keys, h_res_values] = sort_unordered_results(d_keys_out, d_values_out);
  if (g_verbose)
  {
    std::cout << "Output keys:\n";
    DisplayResults(h_res_keys, k);
    std::cout << "\n\n";

    std::cout << "Output values:\n";
    DisplayResults(h_res_values, k);
    std::cout << "\n\n";
  }
  int compare = CompareResults(h_reference_keys_vector.data(), h_res_keys.data(), k, g_verbose);
  AssertEquals(0, compare);
  compare = CompareResults(h_reference_values_vector.data(), h_res_values.data(), k, g_verbose);
  AssertEquals(0, compare);

  std::cout << "\n\n";

  return 0;
}

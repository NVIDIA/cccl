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

#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdio>

#include "../../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

bool g_verbose = false; // Whether to display input/output to console
CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

// Simple key-value pairing for floating point types.
// Treats positive and negative zero as equivalent.
struct Pair
{
  float key;
  int value;

  bool operator<(const Pair& b) const
  {
    bool res = key < b.key;
    if (key == b.key)
    {
      res = value < b.value;
    }
    return res;
  }
};

// Initialize key-value sorting problem.
void Initialize(float* h_keys, int* h_values, float* h_reference_keys, int* h_reference_values, int num_items, int k)
{
  Pair* h_pairs           = new Pair[num_items];
  Pair* h_reference_pairs = new Pair[k];

  for (int i = 0; i < num_items; ++i)
  {
    RandomBits(h_keys[i]);
    RandomBits(h_values[i]);
    h_pairs[i].key   = h_keys[i];
    h_pairs[i].value = h_values[i];
  }

  if (g_verbose)
  {
    printf("Input keys:\n");
    DisplayResults(h_keys, num_items);
    printf("\n\n");

    printf("Input values:\n");
    DisplayResults(h_values, num_items);
    printf("\n\n");
  }

  std::partial_sort_copy(h_pairs, h_pairs + num_items, h_reference_pairs, h_reference_pairs + k);

  for (int i = 0; i < k; ++i)
  {
    h_reference_keys[i]   = h_reference_pairs[i].key;
    h_reference_values[i] = h_reference_pairs[i].value;
  }

  delete[] h_pairs;
  delete[] h_reference_pairs;
}

//  In some case the results of topK is unordered. Sort the results to compare with ground truth.
void SortUnorderedRes(float* h_res_keys, float* d_keys_out, int* h_res_values, int* d_values_out, int k)
{
  CubDebugExit(cudaMemcpy(h_res_keys, d_keys_out, sizeof(float) * k, cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(h_res_values, d_values_out, sizeof(int) * k, cudaMemcpyDeviceToHost));
  Pair* h_res_pairs = new Pair[k];
  for (int i = 0; i < k; ++i)
  {
    h_res_pairs[i].key   = h_res_keys[i];
    h_res_pairs[i].value = h_res_values[i];
  }
  std::stable_sort(h_res_pairs, h_res_pairs + k);
  for (int i = 0; i < k; ++i)
  {
    h_res_keys[i]   = h_res_pairs[i].key;
    h_res_values[i] = h_res_pairs[i].value;
  }
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
    printf("%s "
           "[--n=<input items> "
           "[--k=<output items> "
           "[--device=<device-id>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  printf("cub::DeviceTopK::MinPairs() find %d smallest items from %d items (%d-byte keys %d-byte values)\n",
         k,
         num_items,
         int(sizeof(float)),
         int(sizeof(int)));
  fflush(stdout);

  // Allocate host arrays
  thrust::host_vector<float> h_keys_vector(num_items);
  thrust::host_vector<float> h_reference_keys_vector(k);
  thrust::host_vector<float> h_res_keys_vector(k);
  thrust::host_vector<int> h_values_vector(num_items);
  thrust::host_vector<int> h_reference_values_vector(k);
  thrust::host_vector<int> h_res_values_vector(k);

  float* h_keys           = thrust::raw_pointer_cast(h_keys_vector.data());
  float* h_reference_keys = thrust::raw_pointer_cast(h_reference_keys_vector.data());
  float* h_res_keys       = thrust::raw_pointer_cast(h_res_keys_vector.data());
  int* h_values           = thrust::raw_pointer_cast(h_values_vector.data());
  int* h_reference_values = thrust::raw_pointer_cast(h_reference_values_vector.data());
  int* h_res_values       = thrust::raw_pointer_cast(h_res_values_vector.data());

  // Initialize problem and solution on host
  Initialize(h_keys, h_values, h_reference_keys, h_reference_values, num_items, k);

  // Allocate device arrays
  float* d_keys_in = nullptr;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_keys_in, sizeof(float) * num_items));
  int* d_values_in = nullptr;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_values_in, sizeof(int) * num_items));

  // Initialize device input
  CubDebugExit(cudaMemcpy(d_keys_in, h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_values_in, h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));

  // Allocate device output array and num selected
  float* d_keys_out = nullptr;
  int* d_values_out = nullptr;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_keys_out, sizeof(float) * k));
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_values_out, sizeof(int) * k));

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  void* d_temp_storage      = nullptr;

  // Specify that we do not require a specific output order and do not require deterministic results
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  CubDebugExit(DeviceTopK::MinPairs(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, requirements));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Initialize device arrays
  CubDebugExit(cudaMemcpy(d_keys_in, h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_values_in, h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));

  // Run
  CubDebugExit(DeviceTopK::MinPairs(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, requirements));

  // Check for correctness (and display results, if specified)
  SortUnorderedRes(h_res_keys, d_keys_out, h_res_values, d_values_out, k);
  if (g_verbose)
  {
    printf("Output keys:\n");
    DisplayResults(h_res_keys, k);
    printf("\n\n");

    printf("Output values:\n");
    DisplayResults(h_res_values, k);
    printf("\n\n");
  }
  int compare = CompareResults(h_reference_keys, h_res_keys, k, g_verbose);
  AssertEquals(0, compare);
  compare = CompareResults(h_reference_values, h_res_values, k, g_verbose);
  AssertEquals(0, compare);

  // Cleanup
  if (d_keys_in)
  {
    CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    d_keys_in = nullptr;
  }
  if (d_values_in)
  {
    CubDebugExit(g_allocator.DeviceFree(d_values_in));
    d_values_in = nullptr;
  }
  if (d_keys_out)
  {
    CubDebugExit(g_allocator.DeviceFree(d_keys_out));
    d_keys_out = nullptr;
  }
  if (d_values_out)
  {
    CubDebugExit(g_allocator.DeviceFree(d_values_out));
    d_values_out = nullptr;
  }
  if (d_temp_storage)
  {
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    d_temp_storage = nullptr;
  }

  printf("\n\n");

  return 0;
}

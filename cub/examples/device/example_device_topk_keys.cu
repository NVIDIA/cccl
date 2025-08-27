// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/******************************************************************************
 * Simple example of cub::DeviceTopK::TopKMinKeys().
 *
 * Find the top-k smallest float keys paired with a corresponding array of int values.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_topk_keys.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_topk.cuh>
#include <cub/util_allocator.cuh>

#include <algorithm>

#include "../../test/test_util.h"
#include <stdio.h>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

bool g_verbose = false; // Whether to display input/output to console
CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------
/**
 * Initialize key-value sorting problem.
 */

void Initialize(float* h_keys, float* h_reference_keys, int num_items, int k)
{
  for (int i = 0; i < num_items; ++i)
  {
    RandomBits(h_keys[i]);
  }

  if (g_verbose)
  {
    printf("Input keys:\n");
    DisplayResults(h_keys, num_items);
    printf("\n\n");
  }

  std::partial_sort_copy(h_keys, h_keys + num_items, h_reference_keys, h_reference_keys + k);
}
/**
 *  In some case the results of topK is unordered. Sort the results to compare with groundtruth.
 */
void SortUnorderedRes(float* h_res_keys, float* d_keys_out, int k)
{
  CubDebugExit(cudaMemcpy(h_res_keys, d_keys_out, sizeof(float) * k, cudaMemcpyDeviceToHost));
  std::stable_sort(h_res_keys, h_res_keys + k);
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
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

  printf("cub::DeviceTopK::TopKPairs() find %d largest items from %d items (%d-byte keys %d-byte values)\n",
         k,
         num_items,
         int(sizeof(float)),
         int(sizeof(int)));
  fflush(stdout);

  // Allocate host arrays
  float* h_keys           = new float[num_items];
  float* h_reference_keys = new float[k];
  float* h_res_keys       = new float[k];

  // Initialize problem and solution on host
  Initialize(h_keys, h_reference_keys, num_items, k);

  // Allocate device arrays
  float* d_keys_in = nullptr;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_keys_in, sizeof(float) * num_items));

  // Initialize device input
  CubDebugExit(cudaMemcpy(d_keys_in, h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));

  // Allocate device output array and num selected
  float* d_keys_out = nullptr;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_keys_out, sizeof(float) * k));

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  void* d_temp_storage      = nullptr;

  CubDebugExit(DeviceTopK::TopKMinKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, k));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Initialize device arrays
  CubDebugExit(cudaMemcpy(d_keys_in, h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));

  // Run
  CubDebugExit(DeviceTopK::TopKMinKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, k));

  // Check for correctness (and display results, if specified)
  SortUnorderedRes(h_res_keys, d_keys_out, k);
  int compare = CompareResults(h_reference_keys, h_res_keys, k, g_verbose);
  AssertEquals(0, compare);

  // Cleanup
  if (h_keys)
  {
    delete[] h_keys;
    h_keys = nullptr;
  }
  if (h_reference_keys)
  {
    delete[] h_reference_keys;
    h_reference_keys = nullptr;
  }
  if (h_res_keys)
  {
    delete[] h_res_keys;
    h_res_keys = nullptr;
  }

  if (d_keys_in)
  {
    CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    d_keys_in = nullptr;
  }

  if (d_keys_out)
  {
    CubDebugExit(g_allocator.DeviceFree(d_keys_out));
    d_keys_out = nullptr;
  }

  if (d_temp_storage)
  {
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    d_temp_storage = nullptr;
  }

  printf("\n\n");

  return 0;
}
